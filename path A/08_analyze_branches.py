
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_polyp import resolve_device  # noqa: E402
from utils.dataloader_polyp import get_loader  # noqa: E402
from routing import (  # noqa: E402
    build_model, strip_thop_buffers,
    AdaptiveMultiKernelDepthwiseConv, UncertaintyGuidedMultiKernelDepthwiseConv,
)

ROUTED = (AdaptiveMultiKernelDepthwiseConv, UncertaintyGuidedMultiKernelDepthwiseConv)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--checkpoint', type=str, default='best.pth')
    p.add_argument('--dataset', type=str, default=None)
    p.add_argument('--split', type=str, default='test')
    p.add_argument('--data_root', type=str, default='./data/polyp/target')
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--max_images', type=int, default=40)
    return p.parse_args()


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    run_dir = args.run_dir if os.path.isdir(args.run_dir) else os.path.join(here, 'runs', args.run_dir)
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    if cfg['routing_mode'] not in ('adaptive_soft', 'uncertainty_soft'):
        raise SystemExit('this analysis needs a SOFT-routing run (adaptive_soft / '
                         f'uncertainty_soft); got {cfg["routing_mode"]!r}')

    device = resolve_device(args.device)
    eval_on = args.dataset or cfg['dataset']
    model = build_model(cfg['routing_mode'], network=cfg['network'],
                        kernel_sizes=cfg['kernel_sizes'],
                        deep_supervision=cfg['aux_supervision'],
                        ca_min_squeeze=cfg['ca_min_squeeze'],
                        router_reduction=cfg['router_reduction'],
                        router_temperature=cfg['router_temperature']).to(device)
    model.load_state_dict(strip_thop_buffers(
        torch.load(os.path.join(run_dir, args.checkpoint), map_location=device)), strict=True)
    model.eval()

    ks = cfg['kernel_sizes']
    K = len(ks)
    acc = defaultdict(lambda: {'share': np.zeros(K), 'cos': np.zeros((K, K)),
                               'top1_loss': 0.0, 'drop': np.zeros(K), 'n': 0})

    # Hook every routed block: recompute the branch outputs and the weights it used,
    # then measure how the aggregate decomposes.
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0]
            with torch.no_grad():
                outs = [dw(x) for dw in module.dwconvs]
                w = module.last_routing_weights            # (B, K)
                terms = [w[:, k].view(-1, 1, 1, 1) * outs[k] for k in range(K)]
                agg = sum(terms)
                agg_n = agg.norm().item() + 1e-12

                a = acc[name]
                norms = np.array([t.norm().item() for t in terms])
                a['share'] += norms / (norms.sum() + 1e-12)

                flat = [t.flatten() for t in terms]
                for i in range(K):
                    for j in range(K):
                        a['cos'][i, j] += torch.nn.functional.cosine_similarity(
                            flat[i], flat[j], dim=0).item()

                top = int(w[0].argmax().item())
                a['top1_loss'] += ((terms[top] - agg).norm().item()) / agg_n
                for k in range(K):
                    kept = sum(t for i, t in enumerate(terms) if i != k)
                    a['drop'][k] += ((kept - agg).norm().item()) / agg_n
                a['n'] += 1
        return hook

    for name, m in model.named_modules():
        if isinstance(m, ROUTED):
            handles.append(m.register_forward_hook(make_hook(name)))

    loader = get_loader(image_root=f'{args.data_root}/{eval_on}/{args.split}/images/',
                        gt_root=f'{args.data_root}/{eval_on}/{args.split}/masks/',
                        batchsize=1, trainsize=cfg['img_size'], shuffle=False,
                        split='test', color_image=cfg['color_image'])
    with torch.no_grad():
        for i, (images, _, _, _) in enumerate(loader):
            if i >= args.max_images:
                break
            model(images.to(device))
    for h in handles:
        h.remove()

    print(f'run     : {cfg["run_id"]}')
    print(f'kernels : {ks}   blocks analysed: {len(acc)}   images: {min(args.max_images, i+1)}\n')

    hdr = (f'{"block":<26}' + ''.join(f'{"share k"+str(k):>10}' for k in ks)
           + f'{"top1 loss":>11}' + ''.join(f'{"drop k"+str(k):>10}' for k in ks))
    print(hdr)
    print('-' * len(hdr))

    tot_share = np.zeros(K); tot_drop = np.zeros(K); tot_top1 = 0.0
    tot_cos = np.zeros((K, K)); nb = 0
    for name, a in acc.items():
        n = a['n']
        share, drop = a['share'] / n, a['drop'] / n
        top1 = a['top1_loss'] / n
        tot_share += share; tot_drop += drop; tot_top1 += top1
        tot_cos += a['cos'] / n; nb += 1
        short = name.replace('.multi_scale_dwconv', '')
        print(f'{short:<26}' + ''.join(f'{v:>10.3f}' for v in share)
              + f'{top1:>11.3f}' + ''.join(f'{v:>10.3f}' for v in drop))

    share, drop, top1, cos = tot_share/nb, tot_drop/nb, tot_top1/nb, tot_cos/nb
    print('-' * len(hdr))
    print(f'{"MEAN":<26}' + ''.join(f'{v:>10.3f}' for v in share)
          + f'{top1:>11.3f}' + ''.join(f'{v:>10.3f}' for v in drop))

    print('\nBranch-output cosine similarity (1.0 = identical, 0 = orthogonal):')
    print('        ' + ''.join(f'{"k"+str(k):>8}' for k in ks))
    for i, k in enumerate(ks):
        print(f'   k{k:<4}' + ''.join(f'{cos[i, j]:>8.3f}' for j in range(K)))

    off = [cos[i, j] for i in range(K) for j in range(K) if i != j]
    mean_off = float(np.mean(off))
    print(f'\nmean off-diagonal similarity: {mean_off:.3f}')
    if mean_off < 0.5:
        print('  -> branches are largely COMPLEMENTARY. Dropping two of three (top-1)')
        print('     discards genuinely distinct information, which is why hard routing')
        print('     costs accuracy. A static set that keeps a complementary PAIR is the')
        print('     better pruning target.')
    else:
        print('  -> branches are substantially REDUNDANT; sparsification should be cheap.')

    cheapest = int(np.argmin(drop))
    keep = [k for i, k in enumerate(ks) if i != cheapest]
    print(f'\ntop-1 would lose {top1*100:.1f}% of the aggregate signal.')
    print(f'cheapest single branch to remove: k={ks[cheapest]} '
          f'(costs {drop[cheapest]*100:.1f}%)  ->  suggested static set {keep}')

    RESULTS = os.path.join(here, 'results')
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, 'branch_analysis.json')
    json.dump({'run_id': cfg['run_id'], 'kernels': ks, 'dataset': eval_on,
               'images': int(min(args.max_images, i+1)), 'blocks': nb,
               'mean_share': share.tolist(), 'mean_drop_loss': drop.tolist(),
               'top1_loss': top1, 'cosine': cos.tolist(),
               'mean_off_diagonal_cosine': mean_off,
               'suggested_static_set': keep,
               'per_block': {k: {'share': (v['share']/v['n']).tolist(),
                                 'drop': (v['drop']/v['n']).tolist(),
                                 'top1_loss': v['top1_loss']/v['n']}
                             for k, v in acc.items()}},
              open(out, 'w'), indent=2)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
