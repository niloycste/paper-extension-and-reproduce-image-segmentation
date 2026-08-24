"""
Run: python -W ignore "path A/02_smoke_test.py"
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mkunet_network import MK_UNet  # noqa: E402
from routing import (  # noqa: E402
    binary_entropy, build_model,
    AdaptiveMultiKernelDepthwiseConv, UncertaintyGuidedMultiKernelDepthwiseConv,
    collect_routing_stats, summarize_routing,
)

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"[{'OK' if cond else 'FAIL'}] {name}")


device = torch.device('cpu')
CH_T = [4, 8, 16, 24, 32]
x = torch.randn(2, 3, 352, 352)
y = (torch.rand(2, 1, 352, 352) > 0.7).float()

check('imports succeed (originals import cleanly alongside the extension)', True)

# --- the frozen baseline must be untouched ----------------------------------
n_base_t = sum(p.numel() for p in MK_UNet(channels=CH_T).parameters())
n_base = sum(p.numel() for p in MK_UNet(channels=[16, 32, 64, 96, 160]).parameters())
check(f'baseline MK_UNet_T params unchanged ({n_base_t:,}, expect 27,384)', n_base_t == 27384)
check(f'baseline MK_UNet params unchanged ({n_base/1e6:.4f}M, expect 0.3156M)',
      abs(n_base / 1e6 - 0.3156) < 0.001)
check('build_model("fixed") returns the untouched baseline class',
      type(build_model('fixed', 'MK_UNet_T')).__name__ == 'MK_UNet')

# --- entropy numerical safety ------------------------------------------------
u = binary_entropy(torch.tensor([[-1e4, 0.0, 1e4]]).view(1, 1, 1, 3))
check('entropy finite at saturated logits (no log(0))', torch.isfinite(u).all().item())
check(f'entropy within [0,1] ({u.min().item():.4f}..{u.max().item():.4f})',
      bool((u >= 0).all() and (u <= 1.0 + 1e-6).all()))
check(f'entropy ~1.0 at logit 0 (got {u.flatten()[1].item():.4f})',
      abs(u.flatten()[1].item() - 1.0) < 1e-5)
check(f'entropy ~0.0 when confident (got {u.flatten()[0].item():.1e})',
      u.flatten()[0].item() < 1e-3)

# --- both new models build, run, and train -----------------------------------
sizes = {}
for mode in ('fixed', 'adaptive_soft', 'uncertainty_soft'):
    m = build_model(mode, 'MK_UNet_T', deep_supervision=True).to(device)
    n = sum(p.numel() for p in m.parameters())
    sizes[mode] = n
    m.zero_grad()
    out = m(x)
    loss = sum(w * nn.functional.binary_cross_entropy_with_logits(p, y)
               for w, p in zip([1.0, 0.5, 0.3, 0.2], out))
    loss.backward()
    grads_ok = all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    orphan = sum(1 for _, p in m.named_parameters() if p.requires_grad and p.grad is None)
    check(f'{mode}: 4 heads at full resolution, backward OK, no orphaned params',
          len(out) == 4
          and all(tuple(o.shape) == (2, 1, 352, 352) for o in out)
          and grads_ok and orphan == 0)

print(f"    params  fixed {sizes['fixed']:,} | "
      f"adaptive {sizes['adaptive_soft']:,} (+{100*(sizes['adaptive_soft']-sizes['fixed'])/sizes['fixed']:.1f}%) | "
      f"uncertainty {sizes['uncertainty_soft']:,} "
      f"(+{100*(sizes['uncertainty_soft']-sizes['fixed'])/sizes['fixed']:.1f}%)")
check('Stage B adds only a little over Stage A (the +2 uncertainty inputs)',
      0 < sizes['uncertainty_soft'] - sizes['adaptive_soft'] < 0.02 * sizes['fixed'])

# --- scale-preserving init: identical to the fixed sum at step zero ----------
for cls, name, needs_u in ((AdaptiveMultiKernelDepthwiseConv, 'Stage A', False),
                           (UncertaintyGuidedMultiKernelDepthwiseConv, 'Stage B', True)):
    blk = cls(in_channels=8, kernel_sizes=[1, 3, 5], stride=1).eval()
    xt, ut = torch.randn(1, 8, 16, 16), torch.rand(1, 1, 16, 16)
    with torch.no_grad():
        w0 = blk.router(xt, ut) if needs_u else blk.router(xt)
        plain = sum(dw(xt) for dw in blk.dwconvs)
        routed = blk(xt, ut) if needs_u else blk(xt)
    check(f'{name}: router uniform (~1.0/branch) at init '
          f'({[round(v, 4) for v in w0[0].tolist()]})',
          torch.allclose(w0, torch.ones_like(w0), atol=1e-5))
    check(f'{name}: output == plain fixed sum at init (scale-preserving verified)',
          torch.allclose(routed, plain, atol=1e-5))

# uncertainty must actually reach the routing decision
blk = UncertaintyGuidedMultiKernelDepthwiseConv(in_channels=8, kernel_sizes=[1, 3, 5], stride=1).eval()
xt, ut = torch.randn(1, 8, 16, 16), torch.rand(1, 1, 16, 16)
with torch.no_grad():
    nn.init.normal_(blk.router.fc2.weight, std=0.5)
    w_lo = blk.router(xt, torch.zeros_like(ut))
    w_hi = blk.router(xt, torch.ones_like(ut))
check('uncertainty changes routing once the router is trained (signal is wired in)',
      not torch.allclose(w_lo, w_hi, atol=1e-4))

# --- router placement --------------------------------------------------------
ug = build_model('uncertainty_soft', 'MK_UNet_T').to(device)
ug(x)
n_ug = sum(1 for m in ug.modules() if isinstance(m, UncertaintyGuidedMultiKernelDepthwiseConv))
n_ad = sum(1 for m in ug.modules() if isinstance(m, AdaptiveMultiKernelDepthwiseConv))
check(f'3 uncertainty-guided routers in decoder3/4/5 (got {n_ug})', n_ug == 3)
check(f'7 feature-only routers elsewhere (got {n_ad})', n_ad == 7)

# --- diagnostics --------------------------------------------------------------
summary = summarize_routing(collect_routing_stats(ug))
check(f'collect_routing_stats covers all 10 routers (got {summary["routers"]})',
      summary['routers'] == 10)
check(f'no router collapse at init (entropy {summary["entropy"]:.4f}, expect 1.0)',
      abs(summary['entropy'] - 1.0) < 1e-3)
check('collect_routing_stats returns {} for a plain MK_UNet',
      collect_routing_stats(MK_UNet(channels=CH_T)) == {})

# --- checkpoint round trip -----------------------------------------------------
torch.save(ug.state_dict(), 'ckpt_ext_tmp.pth')
re = build_model('uncertainty_soft', 'MK_UNet_T').to(device)
re.load_state_dict(torch.load('ckpt_ext_tmp.pth', map_location=device))
check('checkpoint save/load round-trips exactly',
      all(torch.equal(a, b) for a, b in zip(ug.state_dict().values(), re.state_dict().values())))
os.remove('ckpt_ext_tmp.pth')

# --- single-kernel control ------------------------------------------------------
k3 = build_model('uncertainty_soft', 'MK_UNet_T', kernel_sizes=[3]).to(device)
with torch.no_grad():
    o3 = k3(torch.randn(1, 3, 352, 352))
check('kernel_sizes=[3] single-kernel control builds and runs',
      tuple(o3[0].shape) == (1, 1, 352, 352))

print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    print('FAILED:', FAIL)
    raise SystemExit(1)
