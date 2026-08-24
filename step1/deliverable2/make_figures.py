
import os
import re
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Derived from this file's own location, so renaming the containing folder
# cannot silently send the figures somewhere else.
FIGDIR = os.path.join(HERE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# --- Palette: the dataviz reference instance, used unchanged (pre-validated).
# Slots 1-3 clear the all-pairs CVD floors in both modes; we never exceed 3 series.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, BASELINE, SURFACE = '#e5e2da', '#c3c2b7', '#ffffff'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'axes.edgecolor': BASELINE,
    'axes.labelcolor': INK2,
    'axes.titlecolor': INK,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'text.color': INK,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'savefig.facecolor': SURFACE,
    'savefig.dpi': 300,
})


def style(ax):
    """Recessive grid and axes; data sits forward of the chrome."""
    ax.grid(axis='y', color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(0.8)


def save_figure(fig, out):
    fig.savefig(out, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(out[:-4] + '.png', bbox_inches='tight', pad_inches=0.04, dpi=300)


# ---------------------------------------------------------------- parse logs
EPOCH_RE = re.compile(r'Epoch: (\d+), Dataset: (test|val), Dice: ([\d.]+), IoU: ([\d.]+)')
FINAL_RE = re.compile(r'Best Val Dice: ([\d.]+)\s+Test Dice at Best Val: ([\d.]+)\s+'
                      r'Total Train Time: ([\d.]+)s')


def parse_log(path):
    """-> (DataFrame of per-epoch metrics, final-summary dict or None)."""
    text = open(path, encoding='utf-8', errors='replace').read()
    rows = [{'epoch': int(e), 'split': s, 'dice': float(d), 'iou': float(i)}
            for e, s, d, i in EPOCH_RE.findall(text)]
    df = pd.DataFrame(rows)
    m = FINAL_RE.search(text)
    final = None
    if m:
        final = {'best_val': float(m.group(1)), 'test_at_best': float(m.group(2)),
                 'train_time_s': float(m.group(3))}
    return df, final


def find_run(dataset):
    """Newest 200-epoch fixed-routing log for `dataset`; (df, final, run_id) or None."""
    pat = os.path.join(ROOT, 'logs', f'train_log_{dataset}_MK_UNet_T_*_e200_*.log')
    best = None
    for path in sorted(glob.glob(pat)):
        if '_rm' in os.path.basename(path):
            continue  # skip adaptive-routing runs; baseline figures only
        df, final = parse_log(path)
        if df.empty:
            continue
        run_id = os.path.basename(path)[len('train_log_'):-len('.log')]
        # prefer a completed run (has a final summary) and the longest history
        key = (final is not None, df['epoch'].max())
        if best is None or key > best[0]:
            best = (key, df, final, run_id)
    return None if best is None else best[1:]


runs = {}
for ds in ('ClinicDB', 'ColonDB'):
    r = find_run(ds)
    if r is not None:
        runs[ds] = r
        df, final, run_id = r
        state = 'complete' if final else f'in progress (epoch {df["epoch"].max()})'
        print(f'{ds:9} {state}  <- {run_id}')
    else:
        print(f'{ds:9} no run found')

# Paper's published MK-UNet-T DICE (Table 1), for reference lines only.
PAPER_T = {'ClinicDB': 91.26, 'ColonDB': 85.03}

# ------------------------------------------------ Fig 1: learning curves
# Job: change over time. Two series (val, test) on ONE axis, same units. Both
# datasets as small multiples on a shared y-scale, so the gap between them is
# read directly off the panels rather than inferred across two separate figures.
# A handful of epochs is noise, not a learning curve -- don't emit a figure that
# invites conclusions the data can't support.
plottable = [(ds, r) for ds, r in runs.items()
             if not r[0].empty and r[0]['epoch'].max() >= 50]
for ds, r in runs.items():
    if (ds, r) not in plottable:
        n = 0 if r[0].empty else r[0]['epoch'].max()
        print(f'skipping {ds} curve: only {n} epochs so far')

if plottable:
    n = len(plottable)
    fig, axes = plt.subplots(1, n, figsize=(3.65 * n, 2.55), sharey=True, squeeze=False)
    for col, (ds, (df, final, run_id)) in enumerate(plottable):
        ax = axes[0][col]
        style(ax)
        val = df[df.split == 'val'].sort_values('epoch')
        test = df[df.split == 'test'].sort_values('epoch')
        ax.plot(test.epoch, test.dice * 100, color=ORANGE, lw=1.2, label='test', zorder=2)
        ax.plot(val.epoch, val.dice * 100, color=BLUE, lw=1.2, label='validation', zorder=3)

        ax.axhline(PAPER_T[ds], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(0.03, 0.96, ds, transform=ax.transAxes, ha='left', va='top',
                fontsize=8.5, color=INK, fontweight='bold',
                bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.7))
        ax.text(0.97, PAPER_T[ds] + 1.4, f'paper {PAPER_T[ds]:.2f}',
                transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                fontsize=7.5, color=MUTED,
                bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.6))

        if final is not None:
            # Mark the checkpoint we actually report -- selected on validation.
            best_ep = int(val.loc[val.dice.idxmax(), 'epoch'])
            ax.plot([best_ep], [final['best_val'] * 100], 'o', ms=5.5, color=BLUE,
                    mec=SURFACE, mew=1.8, zorder=4)
            # Pinned to a clear band below the plateau (axes fraction) rather than
            # a fixed pixel offset, which collided with the curves on ColonDB.
            ax.annotate(f'best val {final["best_val"]*100:.2f} (ep {best_ep})\n'
                        f'-> test {final["test_at_best"]*100:.2f}',
                        xy=(best_ep, final['best_val'] * 100),
                        xytext=(0.97, 0.30), textcoords='axes fraction',
                        ha='right', va='top', fontsize=7.5, color=INK2,
                        arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8,
                                        shrinkA=0, shrinkB=3),
                        bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.8))

        ax.set_xlabel('epoch')
        if col == 0:
            ax.set_ylabel('DICE (%)')
            ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.34),
                      fontsize=7.5, labelcolor=INK2)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max(df.epoch.max(), 10))
    fig.tight_layout()
    out = os.path.join(FIGDIR, 'curves.pdf')
    save_figure(fig, out)
    plt.close(fig)
    print('wrote', out)

# ------------------------------- Fig 2: per-image DICE distribution (ClinicDB)
# Job: distribution -- shows the mean is dragged by a thin tail of hard cases.
xls = glob.glob(os.path.join(ROOT, 'results_polyp', 'Results_*ClinicDB_test.xlsx'))
if xls:
    d = pd.read_excel(sorted(xls)[-1])
    per_img = d[d.Name != 'AVERAGE']
    dice = per_img.Dice.values * 100

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    style(ax)
    counts, _, _ = ax.hist(dice, bins=range(50, 102, 4), color=BLUE,
                           edgecolor=SURFACE, lw=1.2, zorder=2)
    ax.set_ylim(0, max(counts) * 1.18)
    mean = dice.mean()
    ax.axvline(mean, color=ORANGE, lw=1.6, zorder=3)
    ax.text(mean - 1.1, ax.get_ylim()[1] * 0.91, f'mean {mean:.2f}', ha='right',
            fontsize=8, color=ORANGE,
            bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.6))
    n_bad = int((dice < 80).sum())
    ax.text(0.02, 0.92, f'{n_bad} of {len(dice)} images below 80 DICE\n'
                        f'median {pd.Series(dice).median():.2f}   min {dice.min():.2f}',
            transform=ax.transAxes, fontsize=8, color=INK2, va='top',
            bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.8))
    ax.set_xlabel('per-image DICE (%)')
    ax.set_ylabel('images')
    ax.set_xlim(48, 100)
    fig.tight_layout()
    out = os.path.join(FIGDIR, 'dist_clinicdb.pdf')
    save_figure(fig, out)
    plt.close(fig)
    print('wrote', out)

# ------------------------------------- Fig 3: our result vs paper, with variance
# Job: point estimate against a reference that has stated uncertainty.
done = {ds: r for ds, r in runs.items() if r[1] is not None}
if done:
    fig, ax = plt.subplots(figsize=(6.4, 0.95 + 0.62 * len(done)))
    style(ax)
    ax.grid(axis='y', linewidth=0)
    ax.grid(axis='x', color=GRID, linewidth=0.6)
    names = list(done)
    for i, ds in enumerate(names):
        paper = PAPER_T[ds]
        ours = done[ds][1]['test_at_best'] * 100
        # Paper reports 1-4% std dev over its five runs; show the conservative 1%
        # band so the comparison is not read as more precise than it is.
        ax.barh([i], [2 * paper * 0.01], left=[paper - paper * 0.01], height=0.34,
                color=GRID, zorder=1)
        ax.plot([paper], [i], marker='|', ms=16, mew=2, color=MUTED, zorder=2)
        ax.plot([ours], [i], 'o', ms=9, color=BLUE, mec=SURFACE, mew=1.8, zorder=3)
        ax.text(ours, i + 0.30, f'ours {ours:.2f}', ha='center', fontsize=8, color=BLUE,
                bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.6))
        ax.text(paper, i - 0.38, f'paper {paper:.2f}', ha='center', fontsize=8, color=MUTED,
                bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.6))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=INK2)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlabel('DICE (%)')
    lo = min(min(PAPER_T[d] for d in names), min(done[d][1]['test_at_best'] * 100 for d in names))
    hi = max(max(PAPER_T[d] for d in names), max(done[d][1]['test_at_best'] * 100 for d in names))
    ax.set_xlim(lo - 4, hi + 4)
    fig.tight_layout()
    out = os.path.join(FIGDIR, 'vs_paper.pdf')
    save_figure(fig, out)
    plt.close(fig)
    print('wrote', out)

# Note: measured-vs-reported parameter counts are deliberately NOT plotted. The
# five pairs agree to the paper's reported precision, so a chart of overlapping
# points obscures the point and (on a log axis) makes fixed-height markers read as
# error bars the paper never reported. That comparison ships as a table in the
# report instead -- see references/choosing-a-form.md, "is it even a chart?".

print('\nfigures in', FIGDIR)
