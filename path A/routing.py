
import math
import os
import sys
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.helpers import named_apply

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Primitives reused unchanged from the original network definition.
from mkunet_network import (  # noqa: E402
    _init_weights, act_layer, channel_shuffle, gcd,
    ChannelAttention, SpatialAttention, GroupedAttentionGate,
)

__all__ = [
    'binary_entropy',
    'AdaptiveKernelRouter', 'AdaptiveMultiKernelDepthwiseConv',
    'AdaptiveMultiKernelInvertedResidualBlock', 'adaptive_mk_irb_bottleneck',
    'Adaptive_MK_UNet',
    'UncertaintyGuidedKernelRouter', 'UncertaintyGuidedMultiKernelDepthwiseConv',
    'UncertaintyGuidedMKIRBlock', 'ug_mk_irb_stage', 'UG_MK_UNet',
    'SparseMultiKernelDepthwiseConv', 'SparseMKIRBlock', 'sparse_mk_irb_bottleneck',
    'Sparse_MK_UNet', 'reset_branch_calls', 'total_branch_calls', 'set_hard_fraction',
    'collect_routing_stats', 'summarize_routing', 'build_model',
    'strip_thop_buffers',
]


# --------------------------------------------------------------------------
# Stage A — feature-adaptive soft multi-kernel routing
# --------------------------------------------------------------------------

class AdaptiveKernelRouter(nn.Module):
    """Input-conditioned softmax weights over K depth-wise kernel branches.

    weights = K * softmax(logits / T), so with logits at zero (fc2 zero-initialized)
    softmax is uniform (1/K each) and every branch weight starts at exactly 1.0 --
    the aggregation then reproduces the original fixed F1+F3+F5 sum bit-for-bit.
    """
    def __init__(self, in_channels, num_kernels, reduction=4, temperature=1.0):
        super().__init__()
        self.num_kernels = num_kernels
        self.temperature = temperature
        hidden = max(in_channels // reduction, num_kernels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, hidden, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, num_kernels, 1, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        z = self.act(self.fc1(self.pool(x)))
        logits = self.fc2(z).flatten(1) / self.temperature      # (B, K)
        return self.num_kernels * torch.softmax(logits, dim=1)  # sums to K


class AdaptiveMultiKernelDepthwiseConv(nn.Module):
    """Counterpart of MultiKernelDepthwiseConv with a learned weighted sum.

    Returns a single aggregated tensor rather than the list the original returns:
    the weighting is intrinsic to routing and cannot be deferred to the caller the
    way an unweighted sum could.
    """
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6',
                 router_reduction=4, router_temperature=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, k, stride, k // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True)
            )
            for k in kernel_sizes
        ])
        self.router = AdaptiveKernelRouter(in_channels, len(kernel_sizes),
                                           reduction=router_reduction,
                                           temperature=router_temperature)
        self.last_routing_weights = None   # (B, K); read by collect_routing_stats
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
        # named_apply re-initializes every nn.Conv2d it finds, including the
        # router's fc2 -- restore the zero-init so routing starts uniform.
        nn.init.zeros_(self.router.fc2.weight)
        nn.init.zeros_(self.router.fc2.bias)

    def forward(self, x):
        outputs = [dw(x) for dw in self.dwconvs]
        weights = self.router(x)
        self.last_routing_weights = weights.detach()
        dout = 0
        for k, out_k in enumerate(outputs):
            dout = dout + weights[:, k].view(-1, 1, 1, 1) * out_k
        return dout


class AdaptiveMultiKernelInvertedResidualBlock(nn.Module):
    """Counterpart of MultiKernelInvertedResidualBlock.

    Identical expand / channel-shuffle / project / residual structure; only the
    fixed-sum aggregation is replaced. Always produces one weighted tensor, so
    combined_channels == ex_c (the original's add=False concat path is unused by
    every MK_UNet variant and is not reproduced here).
    """
    def __init__(self, in_c, out_c, stride, expansion_factor=2, kernel_sizes=(1, 3, 5),
                 activation='relu6', router_reduction=4, router_temperature=1.0):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.use_skip_connection = (stride == 1)

        self.ex_c = int(in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )
        self.multi_scale_dwconv = AdaptiveMultiKernelDepthwiseConv(
            self.ex_c, kernel_sizes, self.stride, activation,
            router_reduction=router_reduction, router_temperature=router_temperature)
        self.combined_channels = self.ex_c
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_c),
        )
        if self.use_skip_connection and (self.in_c != self.out_c):
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
        # This override runs last in the construction chain, so it is the one that
        # actually survives into the finished module.
        nn.init.zeros_(self.multi_scale_dwconv.router.fc2.weight)
        nn.init.zeros_(self.multi_scale_dwconv.router.fc2.bias)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dout = self.multi_scale_dwconv(pout1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        return out


def adaptive_mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2,
                               kernel_sizes=(1, 3, 5), activation='relu6',
                               router_reduction=4, router_temperature=1.0):
    kw = dict(expansion_factor=expansion_factor, kernel_sizes=kernel_sizes,
              activation=activation, router_reduction=router_reduction,
              router_temperature=router_temperature)
    blocks = [AdaptiveMultiKernelInvertedResidualBlock(in_c, out_c, s, **kw)]
    for _ in range(1, n):
        blocks.append(AdaptiveMultiKernelInvertedResidualBlock(out_c, out_c, 1, **kw))
    return nn.Sequential(*blocks)


# --------------------------------------------------------------------------
# Stage B — uncertainty-guided soft multi-kernel routing
# --------------------------------------------------------------------------

def binary_entropy(logits, eps=1e-7):
    """Normalized predictive entropy of binary logits, in [0, 1].

    0 = confident, 1 = maximally ambiguous. This is PREDICTIVE entropy, not
    Bayesian epistemic uncertainty. Clamping keeps log() away from 0.
    """
    p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / math.log(2.0)


class UncertaintyGuidedKernelRouter(nn.Module):
    """Router conditioned on pooled features AND pooled predictive uncertainty.

        z = concat(GAP(x), mean(U), max(U)) -> MLP -> K logits -> K * softmax(./T)

    Same scale-preserving zero-init as AdaptiveKernelRouter.
    """
    def __init__(self, in_channels, num_kernels, reduction=4, temperature=1.0):
        super().__init__()
        self.num_kernels = num_kernels
        self.temperature = temperature
        hidden = max(in_channels // reduction, num_kernels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels + 2, hidden, 1, bias=True)  # +2: mean/max of U
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, num_kernels, 1, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, u):
        z_feat = self.pool(x)                             # (B, C, 1, 1)
        u_mean = u.mean(dim=(2, 3), keepdim=True)         # (B, 1, 1, 1)
        u_max = u.amax(dim=(2, 3), keepdim=True)          # (B, 1, 1, 1)
        z = torch.cat([z_feat, u_mean, u_max], dim=1)
        logits = self.fc2(self.act(self.fc1(z))).flatten(1) / self.temperature
        return self.num_kernels * torch.softmax(logits, dim=1)


class UncertaintyGuidedMultiKernelDepthwiseConv(nn.Module):
    """Stage-B counterpart of AdaptiveMultiKernelDepthwiseConv. forward(x, u)."""
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6',
                 router_reduction=4, router_temperature=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, k, stride, k // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True)
            )
            for k in kernel_sizes
        ])
        self.router = UncertaintyGuidedKernelRouter(
            in_channels, len(kernel_sizes), reduction=router_reduction,
            temperature=router_temperature)
        self.last_routing_weights = None
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
        nn.init.zeros_(self.router.fc2.weight)
        nn.init.zeros_(self.router.fc2.bias)

    def forward(self, x, u):
        outputs = [dw(x) for dw in self.dwconvs]
        weights = self.router(x, u)
        self.last_routing_weights = weights.detach()
        dout = 0
        for k, out_k in enumerate(outputs):
            dout = dout + weights[:, k].view(-1, 1, 1, 1) * out_k
        return dout


class UncertaintyGuidedMKIRBlock(nn.Module):
    """Stage-B counterpart of AdaptiveMultiKernelInvertedResidualBlock. forward(x, u)."""
    def __init__(self, in_c, out_c, stride, expansion_factor=2, kernel_sizes=(1, 3, 5),
                 activation='relu6', router_reduction=4, router_temperature=1.0):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.use_skip_connection = (stride == 1)

        self.ex_c = int(in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )
        self.multi_scale_dwconv = UncertaintyGuidedMultiKernelDepthwiseConv(
            self.ex_c, kernel_sizes, self.stride, activation,
            router_reduction=router_reduction, router_temperature=router_temperature)
        self.combined_channels = self.ex_c
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_c),
        )
        if self.use_skip_connection and (self.in_c != self.out_c):
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
        nn.init.zeros_(self.multi_scale_dwconv.router.fc2.weight)
        nn.init.zeros_(self.multi_scale_dwconv.router.fc2.bias)

    def forward(self, x, u):
        pout1 = self.pconv1(x)
        dout = self.multi_scale_dwconv(pout1, u)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        return out


class _UGStage(nn.ModuleList):
    """Decoder stage of UncertaintyGuidedMKIRBlocks.

    nn.Sequential cannot forward a second argument, so the stage is an explicit
    ModuleList that threads the uncertainty map through its blocks.
    """
    def forward(self, x, u):
        for block in self:
            x = block(x, u)
        return x


def ug_mk_irb_stage(in_c, out_c, n, s, expansion_factor=2, kernel_sizes=(1, 3, 5),
                    activation='relu6', router_reduction=4, router_temperature=1.0):
    kw = dict(expansion_factor=expansion_factor, kernel_sizes=kernel_sizes,
              activation=activation, router_reduction=router_reduction,
              router_temperature=router_temperature)
    blocks = [UncertaintyGuidedMKIRBlock(in_c, out_c, s, **kw)]
    for _ in range(1, n):
        blocks.append(UncertaintyGuidedMKIRBlock(out_c, out_c, 1, **kw))
    return _UGStage(blocks)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class _RoutedMKUNetBase(nn.Module):
    """Shared skeleton: attention, gates and heads identical to the original MK_UNet.

    Only the multi-kernel aggregation differs; CA / SA / GAG and the four
    segmentation heads are constructed exactly as in the baseline.
    """
    def _build_common(self, num_classes, channels, gag_kernel, ca_min_squeeze):
        self.AG1 = GroupedAttentionGate(F_g=channels[3], F_l=channels[3], F_int=channels[3]//2, kernel_size=gag_kernel, groups=channels[3]//2)
        self.AG2 = GroupedAttentionGate(F_g=channels[2], F_l=channels[2], F_int=channels[2]//2, kernel_size=gag_kernel, groups=channels[2]//2)
        self.AG3 = GroupedAttentionGate(F_g=channels[1], F_l=channels[1], F_int=channels[1]//2, kernel_size=gag_kernel, groups=channels[1]//2)
        self.AG4 = GroupedAttentionGate(F_g=channels[0], F_l=channels[0], F_int=channels[0]//2, kernel_size=gag_kernel, groups=channels[0]//2)

        self.CA1 = ChannelAttention(channels[4], ratio=16, min_squeeze=ca_min_squeeze)
        self.CA2 = ChannelAttention(channels[3], ratio=16, min_squeeze=ca_min_squeeze)
        self.CA3 = ChannelAttention(channels[2], ratio=16, min_squeeze=ca_min_squeeze)
        self.CA4 = ChannelAttention(channels[1], ratio=8, min_squeeze=ca_min_squeeze)
        self.CA5 = ChannelAttention(channels[0], ratio=4, min_squeeze=ca_min_squeeze)
        self.SA = SpatialAttention()

        self.out1 = nn.Conv2d(channels[2], num_classes, kernel_size=1)
        self.out2 = nn.Conv2d(channels[1], num_classes, kernel_size=1)
        self.out3 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.out4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)


class Adaptive_MK_UNet(_RoutedMKUNetBase):
    """Stage A: feature-adaptive soft routing in every encoder and decoder stage.

    CA / SA / GAG and the segmentation heads are byte-identical to MK_UNet; only
    the MKDC aggregation changes.
    """
    def __init__(self, num_classes=1, in_channels=3, channels=(16, 32, 64, 96, 160),
                 depths=(1, 1, 1, 1, 1), kernel_sizes=(1, 3, 5), expansion_factor=2,
                 gag_kernel=3, deep_supervision=False, ca_min_squeeze=1,
                 router_reduction=4, router_temperature=1.0, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        kw = dict(expansion_factor=expansion_factor, kernel_sizes=kernel_sizes,
                  router_reduction=router_reduction, router_temperature=router_temperature)

        self.encoder1 = self._stage(in_channels, channels[0], depths[0], 1, **kw)
        self.encoder2 = self._stage(channels[0], channels[1], depths[1], 1, **kw)
        self.encoder3 = self._stage(channels[1], channels[2], depths[2], 1, **kw)
        self.encoder4 = self._stage(channels[2], channels[3], depths[3], 1, **kw)
        self.encoder5 = self._stage(channels[3], channels[4], depths[4], 1, **kw)

        self.decoder1 = self._stage(channels[4], channels[3], 1, 1, **kw)
        self.decoder2 = self._stage(channels[3], channels[2], 1, 1, **kw)
        self.decoder3 = self._stage(channels[2], channels[1], 1, 1, **kw)
        self.decoder4 = self._stage(channels[1], channels[0], 1, 1, **kw)
        self.decoder5 = self._stage(channels[0], channels[0], 1, 1, **kw)

        self._build_common(num_classes, channels, gag_kernel, ca_min_squeeze)

    def _stage(self, in_c, out_c, n, s, expansion_factor, kernel_sizes,
               router_reduction, router_temperature):
        """Stage builder. Sparse_MK_UNet overrides this to swap in hard routing
        while keeping every parameter name identical."""
        return adaptive_mk_irb_bottleneck(
            in_c, out_c, n, s, expansion_factor=expansion_factor,
            kernel_sizes=kernel_sizes, router_reduction=router_reduction,
            router_temperature=router_temperature)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        out = F.max_pool2d(self.encoder1(x), 2, 2); t1 = out
        out = F.max_pool2d(self.encoder2(out), 2, 2); t2 = out
        out = F.max_pool2d(self.encoder3(out), 2, 2); t3 = out
        out = F.max_pool2d(self.encoder4(out), 2, 2); t4 = out
        out = F.max_pool2d(self.encoder5(out), 2, 2)

        out = self.CA1(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, self.AG1(g=out, x=t4))

        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        p1 = F.interpolate(self.out1(out), scale_factor=(8, 8), mode='bilinear')
        out = torch.add(out, self.AG2(g=out, x=t3))

        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        p2 = F.interpolate(self.out2(out), scale_factor=(4, 4), mode='bilinear')
        out = torch.add(out, self.AG3(g=out, x=t2))

        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        p3 = F.interpolate(self.out3(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, self.AG4(g=out, x=t1))

        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        p4 = self.out4(out)

        if self.deep_supervision:
            return [p4, p3, p2, p1]
        return [p4]


class UG_MK_UNet(_RoutedMKUNetBase):
    """Stage B: uncertainty-guided soft routing.

    Identical to Adaptive_MK_UNet except that decoder3/4/5 route on features AND
    the predictive uncertainty of the immediately preceding (coarser) head.
    Encoder and decoder1/2 stay feature-only -- no earlier prediction exists there.

    Design note: keeping the encoder adaptive in BOTH models (rather than
    fixed-sum here) makes uncertainty the single differing variable, which is what
    the key causal comparison "Adaptive soft + Aux vs UG soft" requires.

    Head naming is the REPOSITORY's: repo p4 is the paper's final p1.

    The intermediate heads only carry meaningful uncertainty when supervised, so
    train this with --deep_supervision True and compare against the matched
    Adaptive + aux control.
    """
    def __init__(self, num_classes=1, in_channels=3, channels=(16, 32, 64, 96, 160),
                 depths=(1, 1, 1, 1, 1), kernel_sizes=(1, 3, 5), expansion_factor=2,
                 gag_kernel=3, deep_supervision=False, ca_min_squeeze=1,
                 router_reduction=4, router_temperature=1.0, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        kw = dict(expansion_factor=expansion_factor, kernel_sizes=kernel_sizes,
                  router_reduction=router_reduction, router_temperature=router_temperature)

        self.encoder1 = adaptive_mk_irb_bottleneck(in_channels, channels[0], depths[0], 1, **kw)
        self.encoder2 = adaptive_mk_irb_bottleneck(channels[0], channels[1], depths[1], 1, **kw)
        self.encoder3 = adaptive_mk_irb_bottleneck(channels[1], channels[2], depths[2], 1, **kw)
        self.encoder4 = adaptive_mk_irb_bottleneck(channels[2], channels[3], depths[3], 1, **kw)
        self.encoder5 = adaptive_mk_irb_bottleneck(channels[3], channels[4], depths[4], 1, **kw)

        # No preceding prediction yet -> feature-only.
        self.decoder1 = adaptive_mk_irb_bottleneck(channels[4], channels[3], 1, 1, **kw)
        self.decoder2 = adaptive_mk_irb_bottleneck(channels[3], channels[2], 1, 1, **kw)
        # Uncertainty-guided, fed by out1 / out2 / out3 respectively.
        self.decoder3 = ug_mk_irb_stage(channels[2], channels[1], 1, 1, **kw)
        self.decoder4 = ug_mk_irb_stage(channels[1], channels[0], 1, 1, **kw)
        self.decoder5 = ug_mk_irb_stage(channels[0], channels[0], 1, 1, **kw)

        self._build_common(num_classes, channels, gag_kernel, ca_min_squeeze)
        self.last_uncertainty = {}

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        out = F.max_pool2d(self.encoder1(x), 2, 2); t1 = out
        out = F.max_pool2d(self.encoder2(out), 2, 2); t2 = out
        out = F.max_pool2d(self.encoder3(out), 2, 2); t3 = out
        out = F.max_pool2d(self.encoder4(out), 2, 2); t4 = out
        out = F.max_pool2d(self.encoder5(out), 2, 2)

        out = self.CA1(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, self.AG1(g=out, x=t4))

        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        logits1 = self.out1(out)                       # native H/8
        p1 = F.interpolate(logits1, scale_factor=(8, 8), mode='bilinear')
        out = torch.add(out, self.AG2(g=out, x=t3))

        # Uncertainty is taken from the PRE-upsample logits, whose native resolution
        # already matches this decoder stage's input -- no interpolation needed.
        # Detached so it acts as a routing signal, not a second gradient path.
        u1 = binary_entropy(logits1).detach()
        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder3(out, u1), scale_factor=(2, 2), mode='bilinear'))
        logits2 = self.out2(out)                       # native H/4
        p2 = F.interpolate(logits2, scale_factor=(4, 4), mode='bilinear')
        out = torch.add(out, self.AG3(g=out, x=t2))

        u2 = binary_entropy(logits2).detach()
        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder4(out, u2), scale_factor=(2, 2), mode='bilinear'))
        logits3 = self.out3(out)                       # native H/2
        p3 = F.interpolate(logits3, scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, self.AG4(g=out, x=t1))

        u3 = binary_entropy(logits3).detach()
        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder5(out, u3), scale_factor=(2, 2), mode='bilinear'))
        p4 = self.out4(out)

        self.last_uncertainty = {'u1_mean': u1.mean().item(),
                                 'u2_mean': u2.mean().item(),
                                 'u3_mean': u3.mean().item()}

        if self.deep_supervision:
            return [p4, p3, p2, p1]
        return [p4]


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def collect_routing_stats(model):
    """Per-router diagnostics from the most recent forward pass.

    Returns {module_name: {...}}, or {} for a model with no routers (e.g. a plain
    MK_UNet) or one that has not run forward() yet. Per router:

      mean_weight  mean weight per branch (weights sum to K, so 1.0 each = uniform,
                   i.e. equivalent to the original fixed sum)
      select_frac  fraction of samples whose argmax picks each branch -- the
                   "percent selecting kernel 1/3/5" collapse check
      entropy      routing entropy normalized to [0,1]: 1.0 = uniform,
                   0.0 = fully collapsed onto one kernel

    A router sitting near entropy 0 has degenerated into a single-kernel model.
    """
    stats = {}
    for name, module in model.named_modules():
        if not isinstance(module, _ROUTED_CONVS):
            continue
        w = module.last_routing_weights
        if w is None:
            continue
        k = w.shape[1]
        p = (w / k).clamp_min(1e-12)
        ent = (-(p * p.log()).sum(dim=1) / math.log(k)).mean().item()
        sel = torch.zeros(k, device=w.device)
        sel.scatter_add_(0, w.argmax(dim=1), torch.ones(w.shape[0], device=w.device))
        stats[name] = {'mean_weight': w.mean(dim=0).tolist(),
                       'select_frac': (sel / w.shape[0]).tolist(),
                       'entropy': ent}
    return stats


def summarize_routing(stats):
    """Collapse per-router stats into a single dict for the training log."""
    if not stats:
        return None
    n = len(stats)
    k = len(next(iter(stats.values()))['mean_weight'])
    mean_w = [sum(s['mean_weight'][i] for s in stats.values()) / n for i in range(k)]
    sel = [sum(s['select_frac'][i] for s in stats.values()) / n for i in range(k)]
    ent = sum(s['entropy'] for s in stats.values()) / n
    return {'routers': n,
            'mean_weight': [round(v, 3) for v in mean_w],
            'select_frac': [round(v, 3) for v in sel],
            'entropy': round(ent, 4)}


# --------------------------------------------------------------------------
# Stage C — sparse / hard top-k multi-kernel routing
#
# Soft routing evaluates every branch and weights the results, so it cannot save
# computation. Sparse routing must actually not execute the unselected branches.
#
# Training vs inference differ, deliberately:
#   training  -- all branches are computed. The gradient wrt a branch weight is
#                <dL/dout, F_k>, which needs every F_k, so skipping during training
#                is impossible with a straight-through estimator. Training is not
#                the deployment scenario, so this costs nothing we care about.
#   inference -- only the selected branch(es) run. Samples are grouped by their
#                selection and each branch is invoked once on its subset; at batch
#                size 1 that is exactly one convolution call. This is the setting
#                any efficiency claim must be measured in.
# --------------------------------------------------------------------------

class SparseMultiKernelDepthwiseConv(nn.Module):
    """Hard top-k routing over the depth-wise branches.

    Parameter names match AdaptiveMultiKernelDepthwiseConv exactly, so a trained
    Stage-A checkpoint loads into a Stage-C model with no key remapping.

    The selected weights are rescaled to sum to K, matching the soft router's
    convention (weights sum to K, ~1.0 each when uniform). Without that rescale,
    switching soft->hard would shrink activations by a factor of K and the
    warm-started model would not match the model it was initialized from.
    """
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6',
                 router_reduction=4, router_temperature=1.0, topk=1):
        super().__init__()
        self.in_channels = in_channels
        self.num_kernels = len(kernel_sizes)
        self.topk = min(topk, self.num_kernels)
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, k, stride, k // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True)
            )
            for k in kernel_sizes
        ])
        self.router = AdaptiveKernelRouter(in_channels, self.num_kernels,
                                           reduction=router_reduction,
                                           temperature=router_temperature)
        self.last_routing_weights = None
        # Counts actual branch invocations, for verifying that skipping is real.
        self.branch_calls = [0] * self.num_kernels
        # Soft-to-hard annealing. 0.0 = pure soft routing (identical to Stage A),
        # 1.0 = fully hard. Ramping this over training avoids the abrupt distribution
        # shift that an immediate switch inflicts on every downstream BatchNorm.
        # Inference always uses hard routing regardless of this value.
        self.hard_frac = 1.0
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
        nn.init.zeros_(self.router.fc2.weight)
        nn.init.zeros_(self.router.fc2.bias)

    def reset_branch_calls(self):
        self.branch_calls = [0] * self.num_kernels

    def _hard_weights(self, w_soft):
        """Top-k weights, rescaled to sum to K; unselected branches exactly zero."""
        topv, topi = w_soft.topk(self.topk, dim=1)
        w_hard = torch.zeros_like(w_soft)
        w_hard.scatter_(1, topi, topv)
        return w_hard * (self.num_kernels / w_hard.sum(dim=1, keepdim=True))

    def forward(self, x):
        w_soft = self.router(x)
        w_hard = self._hard_weights(w_soft)
        self.last_routing_weights = w_hard.detach()

        if self.training:
            # Straight-through: forward uses the hard weights, gradients flow
            # through the soft ones. All branches computed (see module note).
            w_ste = w_hard.detach() + w_soft - w_soft.detach()
            # Blend toward hard over training when annealing is enabled.
            a = self.hard_frac
            w = w_ste if a >= 1.0 else (1.0 - a) * w_soft + a * w_ste
            outs = [dw(x) for dw in self.dwconvs]
            dout = 0
            for k, out_k in enumerate(outs):
                dout = dout + w[:, k].view(-1, 1, 1, 1) * out_k
            return dout

        # Inference: genuinely skip unselected branches.
        if x.shape[0] == 1:
            # Batch-1 fast path -- the deployment setting. Masking a one-element
            # batch would add a gather, an allocation and a scatter to save
            # nothing, so index the selected branches directly.
            dout = None
            for k in torch.nonzero(w_hard[0] > 0, as_tuple=False).flatten().tolist():
                y = self.dwconvs[k](x) * w_hard[0, k]
                self.branch_calls[k] += 1
                dout = y if dout is None else dout + y
            return dout

        # Batched: group samples by selection and invoke each branch once on its
        # subset. Samples choosing different kernels are handled by masking, which
        # is why batch-1 is the honest setting for an efficiency measurement.
        dout = None
        for k, dw in enumerate(self.dwconvs):
            mask = w_hard[:, k] > 0
            if not bool(mask.any()):
                continue                      # branch k is never invoked
            y = dw(x[mask]) * w_hard[mask, k].view(-1, 1, 1, 1)
            self.branch_calls[k] += 1
            if dout is None:
                dout = y.new_zeros((x.shape[0],) + y.shape[1:])
            dout[mask] = dout[mask] + y
        return dout


class SparseMKIRBlock(nn.Module):
    """Stage-C block. Structure identical to the Stage-A block; only the
    aggregation is hard. Parameter names match so checkpoints interchange."""
    def __init__(self, in_c, out_c, stride, expansion_factor=2, kernel_sizes=(1, 3, 5),
                 activation='relu6', router_reduction=4, router_temperature=1.0, topk=1):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.use_skip_connection = (stride == 1)

        self.ex_c = int(in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )
        self.multi_scale_dwconv = SparseMultiKernelDepthwiseConv(
            self.ex_c, kernel_sizes, self.stride, activation,
            router_reduction=router_reduction, router_temperature=router_temperature,
            topk=topk)
        self.combined_channels = self.ex_c
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_c),
        )
        if self.use_skip_connection and (self.in_c != self.out_c):
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
        nn.init.zeros_(self.multi_scale_dwconv.router.fc2.weight)
        nn.init.zeros_(self.multi_scale_dwconv.router.fc2.bias)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dout = self.multi_scale_dwconv(pout1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        return out


def sparse_mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2, kernel_sizes=(1, 3, 5),
                             activation='relu6', router_reduction=4,
                             router_temperature=1.0, topk=1):
    kw = dict(expansion_factor=expansion_factor, kernel_sizes=kernel_sizes,
              activation=activation, router_reduction=router_reduction,
              router_temperature=router_temperature, topk=topk)
    blocks = [SparseMKIRBlock(in_c, out_c, s, **kw)]
    for _ in range(1, n):
        blocks.append(SparseMKIRBlock(out_c, out_c, 1, **kw))
    return nn.Sequential(*blocks)


class Sparse_MK_UNet(Adaptive_MK_UNet):
    """Stage C: hard top-k routing everywhere Stage A used soft routing.

    Subclasses Adaptive_MK_UNet and swaps the stage builder, so the forward pass,
    attention, gates, heads and every parameter name are identical -- a trained
    Stage-A checkpoint loads directly with strict=True.
    """
    def __init__(self, *args, topk=1, **kwargs):
        self._topk = topk
        super().__init__(*args, **kwargs)

    def _stage(self, in_c, out_c, n, s, expansion_factor, kernel_sizes,
               router_reduction, router_temperature):
        return sparse_mk_irb_bottleneck(
            in_c, out_c, n, s, expansion_factor=expansion_factor,
            kernel_sizes=kernel_sizes, router_reduction=router_reduction,
            router_temperature=router_temperature, topk=self._topk)


def set_hard_fraction(model, alpha):
    """Set the soft-to-hard blend on every sparse router (0 = soft, 1 = hard).

    Only affects training; inference is always hard. Returns the number of routers
    updated, so a caller can assert the model actually has any.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, SparseMultiKernelDepthwiseConv):
            m.hard_frac = float(alpha)
            n += 1
    return n


def reset_branch_calls(model):
    """Zero the branch-invocation counters on every sparse router."""
    for m in model.modules():
        if isinstance(m, SparseMultiKernelDepthwiseConv):
            m.reset_branch_calls()


def total_branch_calls(model):
    """(actual invocations, invocations a dense model would make).

    The second number is one call per branch per router -- what the fixed-sum
    baseline and soft routing both do. Their ratio is the honest measure of how
    much depth-wise work was actually skipped.
    """
    actual = dense = 0
    for m in model.modules():
        if isinstance(m, SparseMultiKernelDepthwiseConv):
            actual += sum(m.branch_calls)
            dense += m.num_kernels
    return actual, dense


# Defined here, after every routed-conv class exists, because this tuple is
# evaluated at import time.
_ROUTED_CONVS = (AdaptiveMultiKernelDepthwiseConv,
                 UncertaintyGuidedMultiKernelDepthwiseConv,
                 SparseMultiKernelDepthwiseConv)


def strip_thop_buffers(obj):
    """Remove the `total_ops` / `total_params` buffers that thop.profile injects.

    `cal_params_flops` profiles the model before training and thop leaves these
    buffers attached to every module, so they end up inside the saved state_dict.
    The original `test_polyp.py` works around this by loading with `strict=False`,
    which also silently swallows genuinely missing or mismatched keys -- exactly the
    class of error worth failing loudly on. Stripping them instead lets the load
    stay strict.

    Accepts a model (cleans it in place) or a state_dict (returns a filtered copy).
    """
    _THOP = ('total_ops', 'total_params')
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items()
                if k.split('.')[-1] not in _THOP}
    for module in obj.modules():
        for key in _THOP:
            module._buffers.pop(key, None)
    return obj


NET_CONFIGS = {
    'MK_UNet_T': [4, 8, 16, 24, 32],
    'MK_UNet_S': [8, 16, 32, 48, 80],
    'MK_UNet':   [16, 32, 64, 96, 160],
    'MK_UNet_M': [32, 64, 128, 192, 320],
    'MK_UNet_L': [64, 128, 256, 384, 512],
}


def build_model(routing_mode, network='MK_UNet_T', num_classes=1, in_channels=3,
                kernel_sizes=(1, 3, 5), deep_supervision=False, ca_min_squeeze=1,
                router_reduction=4, router_temperature=1.0, topk=1):
    """Construct the model for a routing mode.

    'fixed' returns the untouched baseline MK_UNet imported from the original
    module, so the baseline arm of every experiment runs the frozen Step-1 code.
    """
    channels = NET_CONFIGS[network]
    common = dict(num_classes=num_classes, in_channels=in_channels, channels=channels,
                  kernel_sizes=list(kernel_sizes), deep_supervision=deep_supervision,
                  ca_min_squeeze=ca_min_squeeze)
    if routing_mode == 'fixed':
        from mkunet_network import MK_UNet
        return MK_UNet(**common)
    if routing_mode == 'adaptive_soft':
        return Adaptive_MK_UNet(router_reduction=router_reduction,
                                router_temperature=router_temperature, **common)
    if routing_mode == 'uncertainty_soft':
        return UG_MK_UNet(router_reduction=router_reduction,
                          router_temperature=router_temperature, **common)
    if routing_mode in ('sparse_top1', 'sparse_top2'):
        return Sparse_MK_UNet(router_reduction=router_reduction,
                              router_temperature=router_temperature,
                              topk=2 if routing_mode == 'sparse_top2' else 1,
                              **common)
    raise ValueError(f'unknown routing_mode {routing_mode!r}')
