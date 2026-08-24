"""
Run: python -W ignore "path A/04_verify_sparse_execution.py"
"""
import json
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing import (  # noqa: E402
    build_model, SparseMultiKernelDepthwiseConv,
    reset_branch_calls, total_branch_calls, summarize_routing, collect_routing_stats,
)

CH_T = 'MK_UNet_T'
device = torch.device('cpu')
PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"[{'OK' if cond else 'FAIL'}] {name}")


# ---------------------------------------------------------------- hook counting
def count_conv_calls(model, x):
    """Actual invocations of each depth-wise branch conv, via forward hooks."""
    calls = {}
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, SparseMultiKernelDepthwiseConv):
            for k, branch in enumerate(mod.dwconvs):
                key = f'{name}.dwconvs.{k}'
                calls[key] = 0

                def hook(m, i, o, key=key):
                    calls[key] += 1
                handles.append(branch.register_forward_hook(hook))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return calls


print('=== 1. does sparse routing really skip branches? ===')
sparse = build_model('sparse_top1', CH_T).to(device).eval()
x1 = torch.randn(1, 3, 352, 352)

calls = count_conv_calls(sparse, x1)
n_routers = sum(1 for m in sparse.modules() if isinstance(m, SparseMultiKernelDepthwiseConv))
executed = sum(1 for v in calls.values() if v > 0)
skipped = sum(1 for v in calls.values() if v == 0)

print(f'    routers: {n_routers} | branch convs: {len(calls)} | '
      f'executed: {executed} | skipped: {skipped}')
check(f'batch-1 top-1 executes exactly 1 branch per router ({executed} == {n_routers})',
      executed == n_routers)
check(f'the other {skipped} branch convs are never called', skipped == len(calls) - n_routers)

# a dense (soft) model must call every branch -- the contrast that makes the above meaningful
soft = build_model('adaptive_soft', CH_T).to(device).eval()
soft_calls = {}
handles = []
for name, mod in soft.named_modules():
    if type(mod).__name__ == 'AdaptiveMultiKernelDepthwiseConv':
        for k, branch in enumerate(mod.dwconvs):
            key = f'{name}.{k}'
            soft_calls[key] = 0

            def h(m, i, o, key=key):
                soft_calls[key] += 1
            handles.append(branch.register_forward_hook(h))
with torch.no_grad():
    soft(x1)
for h in handles:
    h.remove()
check(f'soft routing calls every branch ({sum(1 for v in soft_calls.values() if v > 0)}/'
      f'{len(soft_calls)}) -- so the skipping above is a real difference',
      all(v > 0 for v in soft_calls.values()))

# module's own counters should agree with the hooks
reset_branch_calls(sparse)
with torch.no_grad():
    sparse(x1)
actual, dense = total_branch_calls(sparse)
check(f'internal counters agree with hooks ({actual} calls vs dense {dense})',
      actual == executed and dense == len(calls))
print(f'    depth-wise work executed: {actual}/{dense} = {100*actual/dense:.0f}% of dense')

print('\n=== 2. training must NOT skip (gradients need every branch) ===')
sparse.train()
calls_train = count_conv_calls(sparse, torch.randn(2, 3, 352, 352))
check('in train() mode all branches run (straight-through needs them)',
      all(v > 0 for v in calls_train.values()))
sparse.eval()

print('\n=== 3. measured batch-1 latency ===')


def bench(model, x, warmup=3, iters=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        return (time.perf_counter() - t0) / iters * 1000  # ms


torch.set_num_threads(max(1, os.cpu_count() // 2))
rows = []
for mode in ('fixed', 'adaptive_soft', 'sparse_top1', 'sparse_top2'):
    m = build_model(mode, CH_T).to(device)
    ms = bench(m, x1)
    rows.append((mode, ms, sum(p.numel() for p in m.parameters())))

base_ms = rows[0][1]
print(f'\n{"mode":<16}{"batch-1 latency":>18}{"vs fixed":>12}{"params":>10}')
print('-' * 58)
for mode, ms, n in rows:
    delta = '' if mode == 'fixed' else f'{100*(ms-base_ms)/base_ms:+.1f}%'
    print(f'{mode:<16}{ms:>15.1f} ms{delta:>12}{n:>10,}')

ms = {r[0]: r[1] for r in rows}
# Deliberately NOT a pass/fail check: whether sparse routing is faster is a
# scientific outcome, not a correctness property. Asserting it here would turn a
# hypothesis into a test and hide a negative result. The checks above verify the
# mechanism works; this reports what it buys.
print(f'\n    Executed depth-wise work: {actual}/{dense} = {100*actual/dense:.0f}% of dense.')
if ms['sparse_top1'] < ms['fixed']:
    print(f'    -> sparse top-1 IS faster than the fixed baseline '
          f'({ms["sparse_top1"]:.1f} vs {ms["fixed"]:.1f} ms).')
else:
    print(f'    -> sparse top-1 is NOT faster than the fixed baseline '
          f'({ms["sparse_top1"]:.1f} vs {ms["fixed"]:.1f} ms) despite executing '
          f'{100*actual/dense:.0f}% of the\n       depth-wise work. Router overhead '
          f'(a full-map global pool per block) and per-call\n       dispatch cost exceed '
          f'the saved convolutions at this model size. Report this\n       honestly: FLOP '
          f'reduction without latency reduction is not a speedup.')

# ---------------------------------------------------------------- persist
# These numbers are the evidence behind the efficiency claim, so they are written
# to disk rather than left in console scrollback.
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS, exist_ok=True)
record = {
    'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
    'host': {'platform': platform.platform(), 'processor': platform.processor(),
             'torch': torch.__version__, 'cuda': torch.cuda.is_available(),
             'torch_threads': torch.get_num_threads()},
    'network': CH_T,
    'conditional_execution': {
        'routers': n_routers, 'branch_convs': len(calls),
        'executed_at_batch1': executed, 'skipped_at_batch1': skipped,
        'executed_fraction': round(actual / dense, 4),
        'all_branches_run_in_train_mode': all(v > 0 for v in calls_train.values()),
    },
    'latency_batch1_ms': {mode: round(v, 2) for mode, v, _ in rows},
    'params': {mode: n for mode, _, n in rows},
    'latency_vs_fixed_pct': {mode: round(100 * (v - base_ms) / base_ms, 2)
                             for mode, v, _ in rows},
    'checks_passed': len(PASS), 'checks_failed': len(FAIL),
    'note': ('Latency is the basis of any efficiency claim; executed-branch count is '
             'necessary but not sufficient, since dynamic control flow has its own cost.'),
}
out = os.path.join(RESULTS, 'sparse_verification.json')
json.dump(record, open(out, 'w'), indent=2)
print(f'\nwrote {out}')

print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    print('FAILED:', FAIL)
    raise SystemExit(1)
