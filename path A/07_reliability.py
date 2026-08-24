
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_polyp import resolve_device  # noqa: E402
from utils.dataloader_polyp import get_loader  # noqa: E402
from routing import build_model, strip_thop_buffers, binary_entropy  # noqa: E402

# dataviz reference palette (pre-validated); print/report figures, light surface only
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, BASELINE, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--checkpoint', type=str, default='best.pth')
    p.add_argument('--dataset', type=str, default=None, choices=['ClinicDB', 'ColonDB'])
    p.add_argument('--split', type=str, default='test')
    p.add_argument('--data_root', type=str, default='./data/polyp/target')
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--bins', type=int, default=15)
    p.add_argument('--auroc_sample', type=int, default=2_000_000,
                   help='pixels subsampled for AUROC/correlation (all pixels used for ECE/Brier)')
    return p.parse_args()


def expected_calibration_error(conf, correct, n_bins=15):
    """ECE plus the per-bin data needed for a reliability diagram."""
    edges = np.linspace(0.5, 1.0, n_bins + 1)   # binary confidence lives in [0.5, 1]
    ece, rows = 0.0, []
    n = len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            rows.append((0.5 * (lo + hi), np.nan, np.nan, 0))
            continue
        acc, avg_conf = float(correct[m].mean()), float(conf[m].mean())
        ece += (cnt / n) * abs(acc - avg_conf)
        rows.append((0.5 * (lo + hi), acc, avg_conf, cnt))
    return ece, rows


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    run_dir = args.run_dir if os.path.isdir(args.run_dir) else os.path.join(here, 'runs', args.run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f'run directory not found: {args.run_dir}')

    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    device = resolve_device(args.device)
    eval_on = args.dataset or cfg['dataset']
    cross = eval_on != cfg['dataset']

    model = build_model(cfg['routing_mode'], network=cfg['network'],
                        kernel_sizes=cfg['kernel_sizes'],
                        deep_supervision=cfg['aux_supervision'],
                        ca_min_squeeze=cfg['ca_min_squeeze'],
                        router_reduction=cfg['router_reduction'],
                        router_temperature=cfg['router_temperature']).to(device)
    state = strip_thop_buffers(torch.load(os.path.join(run_dir, args.checkpoint),
                                          map_location=device))
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f'run       : {cfg["run_id"]}')
    print(f'routing   : {cfg["routing_mode"]} | aux {cfg["aux_supervision"]}')
    print(f'evaluating: {eval_on}/{args.split}' + ('  [CROSS-DATASET]' if cross else ''))

    data_path = f'{args.data_root}/{eval_on}/'
    loader = get_loader(image_root=f'{data_path}/{args.split}/images/',
                        gt_root=f'{data_path}/{args.split}/masks/',
                        batchsize=1, trainsize=cfg['img_size'], shuffle=False,
                        split='test', color_image=cfg['color_image'])

    probs, gts, ents, per_image = [], [], [], []
    with torch.no_grad():
        for images, gt, _, names in loader:
            images = images.to(device)
            logits = model(images)[0]                       # final head, always supervised
            gt_r = F.interpolate(gt.to(device).float(), size=logits.shape[-2:],
                                 mode='nearest')
            p = torch.sigmoid(logits)
            u = binary_entropy(logits)

            pf = p.flatten().cpu().numpy()
            gf = (gt_r.flatten() >= 0.5).cpu().numpy().astype(np.uint8)
            uf = u.flatten().cpu().numpy()
            probs.append(pf); gts.append(gf); ents.append(uf)

            err = ((pf >= 0.5).astype(np.uint8) != gf)
            per_image.append({'name': names[0], 'err_rate': float(err.mean()),
                              'mean_entropy': float(uf.mean())})

    probs = np.concatenate(probs); gts = np.concatenate(gts); ents = np.concatenate(ents)
    pred = (probs >= 0.5).astype(np.uint8)
    correct = (pred == gts)
    conf = np.maximum(probs, 1.0 - probs)
    err = (~correct).astype(np.uint8)

    ece, bins = expected_calibration_error(conf, correct, args.bins)
    brier = float(np.mean((probs - gts) ** 2))

    # AUROC / correlation on a subsample -- full-resolution pixel counts are large
    # and these two statistics are stable under sampling.
    rng = np.random.default_rng(0)
    idx = rng.choice(len(err), size=min(args.auroc_sample, len(err)), replace=False)
    if err[idx].sum() == 0 or err[idx].sum() == len(idx):
        auroc, corr = float('nan'), float('nan')
    else:
        auroc = float(roc_auc_score(err[idx], ents[idx]))
        corr = float(np.corrcoef(ents[idx], err[idx])[0, 1])

    print(f'\npixels analysed : {len(err):,}   error rate {err.mean()*100:.2f}%')
    print(f'ECE             : {ece:.4f}   (0 = perfectly calibrated)')
    print(f'Brier           : {brier:.4f}   (lower is better)')
    print(f'error AUROC     : {auroc:.4f}   (0.5 = entropy carries no information about error)')
    print(f'entropy-error r : {corr:.4f}')
    if not np.isnan(auroc):
        if auroc > 0.8:
            print('  -> entropy is strongly associated with error; usable as a difficulty signal.')
        elif auroc > 0.65:
            print('  -> entropy carries a usable but moderate error signal.')
        else:
            print('  -> WEAK association. Using this as a routing signal is not justified '
                  'by this evidence.')

    # correlation across images (does a harder image look more uncertain overall?)
    ir = np.array([d['err_rate'] for d in per_image])
    iu = np.array([d['mean_entropy'] for d in per_image])
    img_corr = float(np.corrcoef(iu, ir)[0, 1]) if len(ir) > 2 else float('nan')
    print(f'per-image r     : {img_corr:.4f}   (image-level entropy vs error rate)')

    tag = f'{eval_on}_{args.split}' + ('_cross' if cross else '')
    out = {'run_id': cfg['run_id'], 'routing_mode': cfg['routing_mode'],
           'aux_supervision': cfg['aux_supervision'], 'evaluated_on': eval_on,
           'split': args.split, 'cross_dataset': cross, 'checkpoint': args.checkpoint,
           'n_pixels': int(len(err)), 'pixel_error_rate': float(err.mean()),
           'ece': ece, 'brier': brier, 'error_auroc': auroc,
           'entropy_error_corr': corr, 'per_image_corr': img_corr,
           'reliability_bins': [{'conf_mid': float(a), 'accuracy': None if np.isnan(b) else float(b),
                                 'avg_conf': None if np.isnan(c) else float(c), 'count': int(d)}
                                for a, b, c, d in bins]}
    jpath = os.path.join(run_dir, f'reliability_{tag}.json')
    json.dump(out, open(jpath, 'w'), indent=2)

    # ---------------- reliability diagram + image-level scatter ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax in (ax1, ax2):
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_linewidth(0.8)

    mids = [b[0] for b in bins]
    accs = [b[1] for b in bins]
    ax1.plot([0.5, 1], [0.5, 1], ls=(0, (4, 3)), lw=1.0, color=MUTED)
    ax1.plot(mids, accs, marker='o', ms=5, lw=1.6, color=BLUE, mec=SURFACE, mew=1.2)
    ax1.set_xlabel('confidence'); ax1.set_ylabel('accuracy')
    ax1.set_title(f'Reliability diagram  (ECE {ece:.3f})', fontsize=10, loc='left', pad=8)
    ax1.set_xlim(0.5, 1.0); ax1.set_ylim(0.5, 1.005)
    ax1.text(0.52, 0.53, 'above line = under-confident\nbelow = over-confident',
             fontsize=7.5, color=MUTED)

    ax2.scatter(iu, ir * 100, s=22, color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax2.set_xlabel('mean predictive entropy'); ax2.set_ylabel('pixel error rate (%)')
    ax2.set_title(f'Per-image: entropy vs error  (r = {img_corr:.2f})',
                  fontsize=10, loc='left', pad=8)
    fig.tight_layout()
    fpath = os.path.join(run_dir, f'reliability_{tag}.pdf')
    fig.savefig(fpath, bbox_inches='tight')
    fig.savefig(fpath[:-4] + '.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

    print(f'\nwrote {jpath}')
    print(f'wrote {fpath}')


if __name__ == '__main__':
    main()
