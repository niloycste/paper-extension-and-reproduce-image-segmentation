"""
Run: python -W ignore "path A/03_measure_cost.py"
"""
import json
import os
import sys

import torch
from thop import profile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing import build_model  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS, exist_ok=True)

NET = 'MK_UNet_T'
x256 = torch.randn(1, 3, 256, 256)      # the paper's FLOP reporting resolution

MODES = [
    ('fixed', 'MK-UNet-T baseline (fixed sum)'),
    ('adaptive_soft', 'Stage A  adaptive soft'),
    ('uncertainty_soft', 'Stage B  uncertainty-guided'),
    ('sparse_top1', 'Stage C  sparse top-1'),
]

print(f'{"arm":<34}{"params":>10}{"vs base":>10}{"FLOPs@256":>13}{"vs base":>10}')
print('-' * 79)
rows, base_p, base_f = [], None, None
for mode, label in MODES:
    m = build_model(mode, NET).eval()
    p = sum(q.numel() for q in m.parameters())
    f, _ = profile(m, inputs=(x256,), verbose=False)
    if base_p is None:
        base_p, base_f = p, f
        dp = df = ''
    else:
        dp, df = f'{100*(p-base_p)/base_p:+.1f}%', f'{100*(f-base_f)/base_f:+.2f}%'
    print(f'{label:<34}{p:>10,}{dp:>10}{f/1e9:>12.6f}G{df:>10}')
    rows.append({'mode': mode, 'label': label, 'params': p, 'flops_g_at256': f / 1e9,
                 'params_vs_base_pct': None if base_p == p else round(100*(p-base_p)/base_p, 2),
                 'flops_vs_base_pct': None if base_f == f else round(100*(f-base_f)/base_f, 2)})

# thop counts convolutions; it does not count the per-sample elementwise weighting,
# which is memory-bandwidth bound. Measured latency (04_verify_sparse_execution.py) is therefore
# the number an efficiency claim should rest on, not this table.
REFERENCE = [
    ('MK-UNet-T (paper)', 0.027, 0.062), ('MK-UNet-S (paper)', 0.093, 0.125),
    ('MK-UNet (paper)', 0.316, 0.314), ('UltraLight VM-UNet', 0.050, 0.060),
    ('EGE-UNet', 0.054, 0.072), ('UNeXt', 1.470, 0.570), ('TransUNet', 105.32, 38.52),
]
print(f'\n{"published reference (paper Table 1)":<34}{"params":>10}{"":>10}{"FLOPs":>13}')
print('-' * 79)
for n, p, f in REFERENCE:
    print(f'{n:<34}{p:>9.3f}M{"":>10}{f:>12.3f}G')

out = os.path.join(RESULTS, 'cost_table.json')
json.dump({'network': NET, 'flop_resolution': 256, 'arms': rows,
           'published_reference': [{'name': n, 'params_m': p, 'flops_g': f}
                                   for n, p, f in REFERENCE],
           'note': ('thop does not count the elementwise branch weighting, which is '
                    'memory-bandwidth bound; see results/sparse_verification.json for '
                    'measured latency.')},
          open(out, 'w'), indent=2)
print(f'\nwrote {out}')
