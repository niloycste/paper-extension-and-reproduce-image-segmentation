
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(HERE, 'runs')
RESULTS = os.path.join(HERE, 'results')
PY = sys.executable

ap = argparse.ArgumentParser()
ap.add_argument('--force', action='store_true', help='redo analysis even if present')
ap.add_argument('--skip_sparse_bench', action='store_true',
                help='skip the latency benchmark (it needs an idle machine)')
args = ap.parse_args()

os.makedirs(RESULTS, exist_ok=True)


def run(cmd, label):
    print(f'\n--- {label} ---')
    r = subprocess.run([PY, '-W', 'ignore'] + cmd, cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f'  FAILED ({r.returncode})')
        print('  ' + (r.stderr or '').strip()[-600:])
        return False
    tail = [l for l in (r.stdout or '').splitlines()
            if l.strip() and 'it/s' not in l and not l.startswith('[INFO]')]
    for l in tail[-6:]:
        print('  ' + l)
    return True


# ---------------------------------------------------------------- per-run
completed, incomplete = [], []
for d in sorted(glob.glob(os.path.join(RUNS, '*'))):
    if not os.path.isfile(os.path.join(d, 'config.json')):
        continue
    (completed if os.path.isfile(os.path.join(d, 'result.json')) else incomplete).append(d)

print(f'{len(completed)} completed run(s), {len(incomplete)} incomplete (skipped)')
for d in incomplete:
    print(f'  incomplete: {os.path.basename(d)}')

for d in completed:
    name = os.path.basename(d)
    cfg = json.load(open(os.path.join(d, 'config.json')))
    # A 1-epoch pipeline test is not a result; do not spend analysis on it.
    if cfg['epoch'] < 5:
        print(f'\nskipping {name} (only {cfg["epoch"]} epoch(s) -- a pipeline test, '
              f'not a result)')
        continue

    if args.force or not glob.glob(os.path.join(d, 'test_metrics_*.json')):
        run(['path A/06_evaluate.py', '--run_dir', d, '--device', 'cpu'],
            f'evaluate {name}')
    else:
        print(f'\nevaluate {name}: already done')

    if args.force or not glob.glob(os.path.join(d, 'reliability_*.json')):
        run(['path A/07_reliability.py', '--run_dir', d, '--device', 'cpu'],
            f'reliability {name}')
    else:
        print(f'reliability {name}: already done')

# ---------------------------------------------------------------- global
run(['path A/03_measure_cost.py'], 'params / FLOPs table')
if not args.skip_sparse_bench:
    run(['path A/04_verify_sparse_execution.py'], 'sparse verification + batch-1 latency')
run(['path A/10_collect_results.py'], 'collect all runs')

# ---------------------------------------------------------------- figure
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    BLUE, ORANGE, MUTED = '#2a78d6', '#eb6834', '#898781'
    GRID, SURFACE, INK2 = '#e5e2da', '#ffffff', '#52514e'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                         'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2,
                         'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False,
                         'savefig.facecolor': SURFACE,
                         'savefig.dpi': 300})

    rows = []
    for d in completed:
        cfg = json.load(open(os.path.join(d, 'config.json')))
        if cfg['epoch'] < 5:
            continue
        res = json.load(open(os.path.join(d, 'result.json')))
        rows.append({'label': f"{cfg['routing_mode']}\nk{''.join(map(str,cfg['kernel_sizes']))}"
                              f"{' +aux' if cfg['aux_supervision'] else ''}\n{cfg['epoch']}ep",
                     'dice': res['test_dice_at_best_val'] * 100,
                     'params': cfg['params'], 'epochs': cfg['epoch']})
    if rows:
        rows.sort(key=lambda r: r['dice'])
        fig, ax = plt.subplots(figsize=(max(6.4, 1.15 * len(rows)), 3.05))
        ax.grid(axis='y', color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        xs = range(len(rows))
        cols = [ORANGE if r['epochs'] < 150 else BLUE for r in rows]
        ax.bar(xs, [r['dice'] for r in rows], color=cols, width=0.62,
               edgecolor=SURFACE, linewidth=0.8)
        for i, r in enumerate(rows):
            ax.text(i, r['dice'] + 0.6, f"{r['dice']:.2f}", ha='center',
                    fontsize=8, color=INK2,
                    bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.6))
        ax.axhline(91.26, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        # Keep the reference label away from the highest result labels at the
        # right side of the chart.
        ax.text(0.02, 91.9, 'paper 91.26', ha='left', fontsize=8, color=MUTED,
                bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.8))
        ax.set_xticks(list(xs))
        ax.set_xticklabels([r['label'] for r in rows], fontsize=7.5, color=INK2)
        ax.set_ylabel('test DICE (%)')
        ax.set_ylim(0, 100)
        fig.tight_layout()
        out = os.path.join(RESULTS, 'summary_figure.pdf')
        fig.savefig(out, bbox_inches='tight', pad_inches=0.04)
        fig.savefig(out[:-4] + '.png', bbox_inches='tight', pad_inches=0.04, dpi=300)
        plt.close(fig)
        print(f'\nwrote {out}')
except Exception as e:                                    # noqa: BLE001
    print(f'\nsummary figure skipped: {e}')

print('\n' + '=' * 60)
print('Finalisation complete. Review:')
print('  path A/results/all_runs.csv          every run, one table')
print('  path A/results/summary_figure.pdf    DICE by arm')
print('  path A/results/cost_table.json       params + FLOPs')
print('  path A/results/branch_analysis.json  branch analysis')
print('=' * 60)
