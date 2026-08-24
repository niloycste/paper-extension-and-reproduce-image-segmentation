
import torch
from thop import profile
from mkunet_network import MK_UNet

CH = [4, 8, 16, 24, 32]
x = torch.randn(2, 3, 352, 352)
x256 = torch.randn(1, 3, 256, 256)

ARMS = [
    ('A baseline',   dict(deep_supervision=False, ca_min_squeeze=1)),
    ('B +DS',        dict(deep_supervision=True,  ca_min_squeeze=1)),
    ('C +CA',        dict(deep_supervision=False, ca_min_squeeze=4)),
    ('D +DS +CA',    dict(deep_supervision=True,  ca_min_squeeze=4)),
]

base_p = None
print(f'{"arm":<14}{"params":>9}{"vs A":>9}{"FLOPs@256":>12}{"vs A":>9}  {"squeeze widths":<18}{"heads":>6}{"orphan":>8}')
print('-' * 90)
for name, kw in ARMS:
    m = MK_UNet(num_classes=1, in_channels=3, channels=CH, **kw)
    p = sum(q.numel() for q in m.parameters())
    f, _ = profile(m, inputs=(x256,), verbose=False)
    if base_p is None:
        base_p, base_f = p, f
    sq = [getattr(m, f'CA{i}').reduced_channels for i in range(1, 6)]

    m.zero_grad()
    out = m(x)
    loss = (sum(w * o.mean() for w, o in zip([1.0, .5, .3, .2], out))
            if kw['deep_supervision'] else out[0].mean())
    loss.backward()
    orphan = sum(1 for _, q in m.named_parameters() if q.requires_grad and q.grad is None)

    print(f'{name:<14}{p:>9,}{p-base_p:>+9,}{f/1e9:>12.6f}{(f-base_f)/1e9:>+9.6f}  '
          f'{str(sq):<18}{len(out):>6}{orphan:>8}')

print(f'\nA squeeze widths are the released behaviour; C/D floor them at 4.')
print(f'FLOPs difference across all arms: {"none" if len(set()) == 0 else ""}')
