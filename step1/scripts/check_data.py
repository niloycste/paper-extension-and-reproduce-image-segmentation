
import os
import sys

EXPECTED = {
    'ClinicDB': {'train': 489, 'val': 61, 'test': 62},
    'ColonDB':  {'train': 303, 'val': 38, 'test': 38},
}
ROOT = sys.argv[1] if len(sys.argv) > 1 else 'data/polyp/target'

ok = True
for ds, splits in EXPECTED.items():
    print(f'\n{ds}  ({ROOT}/{ds})')
    if not os.path.isdir(os.path.join(ROOT, ds)):
        print('  MISSING - dataset folder not found')
        ok = False
        continue
    total = 0
    for split, expect_n in splits.items():
        img_d = os.path.join(ROOT, ds, split, 'images')
        gt_d = os.path.join(ROOT, ds, split, 'masks')
        if not (os.path.isdir(img_d) and os.path.isdir(gt_d)):
            print(f'  {split:5}  MISSING images/ or masks/')
            ok = False
            continue
        n_i, n_m = len(os.listdir(img_d)), len(os.listdir(gt_d))
        total += n_i
        status = 'OK' if (n_i == n_m == expect_n) else f'EXPECTED {expect_n}'
        if status != 'OK':
            ok = False
        print(f'  {split:5}  images={n_i:4}  masks={n_m:4}   {status}')
    print(f'  total images = {total}')

print('\nAll datasets present and correctly split.' if ok else '\nPROBLEMS FOUND - see above.')
sys.exit(0 if ok else 1)
