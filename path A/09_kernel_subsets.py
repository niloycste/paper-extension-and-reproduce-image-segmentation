
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from routing import build_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
os.makedirs(RESULTS, exist_ok=True)

NET = 'MK_UNet_T'
FULL = [1, 3, 5]

base = sum(p.numel() for p in build_model('fixed', NET, kernel_sizes=FULL).parameters())

# Per-branch signal statistics measured on the trained Stage-A model.
bp = os.path.join(RESULTS, 'branch_analysis.json')
ba = json.load(open(bp)) if os.path.isfile(bp) else None
share = dict(zip(FULL, ba['mean_share'])) if ba else {}
drop_loss = dict(zip(FULL, ba['mean_drop_loss'])) if ba else {}

subsets = []
for r in range(1, len(FULL) + 1):
    for c in itertools.combinations(FULL, r):
        subsets.append(list(c))

print(f'{NET}: full [1,3,5] = {base:,} parameters\n')
print(f'{"kernels":<12}{"params":>10}{"saved":>10}{"saved %":>10}'
      f'{"signal kept":>13}{"cost of drop":>14}')
print('-' * 69)

rows = []
for ks in subsets:
    m = build_model('fixed', NET, kernel_sizes=ks)
    p = sum(q.numel() for q in m.parameters())
    saved = base - p
    dropped = [k for k in FULL if k not in ks]
    kept_share = sum(share.get(k, 0.0) for k in ks) if share else None
    # relative aggregate error if exactly one branch is removed (measured)
    cost = drop_loss.get(dropped[0]) if len(dropped) == 1 else None
    print(f'{str(ks):<12}{p:>10,}{saved:>10,}{100*saved/base:>9.1f}%'
          f'{("--" if kept_share is None else f"{kept_share:.3f}"):>13}'
          f'{("--" if cost is None else f"{cost:.3f}"):>14}')
    rows.append({'kernels': ks, 'params': p, 'saved': saved,
                 'saved_pct': round(100 * saved / base, 2),
                 'signal_share_kept': kept_share, 'drop_cost': cost})

if share:
    print('\nParameter cost vs signal carried, per branch:')
    print(f'{"kernel":<9}{"param units":>13}{"signal share":>15}{"share per unit":>17}')
    print('-' * 54)
    for k in FULL:
        units = k * k + 2                       # weight k^2 + BN scale/shift, per channel
        print(f'{k:<9}{units:>13}{share[k]:>15.3f}{share[k]/units:>17.4f}')
    print('\n  -> 1x1 is by far the most parameter-efficient branch; 5x5 is the least,')
    print('     yet 5x5 carries the largest signal share. Removing the branch that')
    print('     saves the most parameters is therefore NOT the same as removing the')
    print('     one that costs the least accuracy -- which is why this needs measuring.')

out = os.path.join(RESULTS, 'kernel_subsets.json')
json.dump({'network': NET, 'full_kernels': FULL, 'base_params': base,
           'subsets': rows,
           'per_branch': ({str(k): {'param_units': k*k+2, 'signal_share': share[k],
                                    'share_per_unit': share[k]/(k*k+2)} for k in FULL}
                          if share else None),
           'note': ('signal_share_kept and drop_cost come from branch_analysis.json, '
                    'measured on the trained Stage-A model; they are norms, not DICE. '
                    'A large norm change may still fine-tune back.')},
          open(out, 'w'), indent=2)
print(f'\nwrote {out}')
