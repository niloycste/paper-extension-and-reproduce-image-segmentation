
import argparse
import json
import os
import sys
from types import SimpleNamespace

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Metric + inference code reused unchanged from the original evaluation script.
from test_polyp import test as run_eval  # noqa: E402
from train_polyp import resolve_device  # noqa: E402
from utils.dataloader_polyp import get_loader  # noqa: E402
from routing import (build_model, collect_routing_stats,  # noqa: E402
                               summarize_routing, strip_thop_buffers)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True,
                   help='run directory produced by 05_train.py (contains config.json)')
    p.add_argument('--checkpoint', type=str, default='best.pth',
                   choices=['best.pth', 'last.pth'])
    p.add_argument('--dataset', type=str, default=None, choices=['ClinicDB', 'ColonDB'],
                   help='override the training dataset to measure cross-dataset generalization')
    p.add_argument('--split', type=str, default='test', choices=['test', 'val'])
    p.add_argument('--data_root', type=str, default='./data/polyp/target')
    p.add_argument('--test_batchsize', type=int, default=1)
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--save_predictions', type=lambda v: str(v).lower() != 'false', default=True)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir if os.path.isdir(args.run_dir) else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', args.run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f'run directory not found: {args.run_dir}')

    with open(os.path.join(run_dir, 'config.json')) as f:
        cfg = json.load(f)

    device = resolve_device(args.device)
    trained_on = cfg['dataset']
    eval_on = args.dataset or trained_on
    cross = eval_on != trained_on

    # Rebuild exactly as trained -- every architectural setting comes from config.json.
    model = build_model(cfg['routing_mode'],
                        network=cfg['network'],
                        kernel_sizes=cfg['kernel_sizes'],
                        deep_supervision=cfg['aux_supervision'],
                        ca_min_squeeze=cfg['ca_min_squeeze'],
                        router_reduction=cfg['router_reduction'],
                        router_temperature=cfg['router_temperature']).to(device)

    ckpt = os.path.join(run_dir, args.checkpoint)
    state = strip_thop_buffers(torch.load(ckpt, map_location=device))
    # strict=True on purpose: config.json guarantees the architecture matches, so a
    # key mismatch here is a real error and should not be silently ignored.
    model.load_state_dict(state, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    print(f'run        : {cfg["run_id"]}')
    print(f'routing    : {cfg["routing_mode"]} | kernels {cfg["kernel_sizes"]} | '
          f'aux {cfg["aux_supervision"]} | params {n_params:,}')
    print(f'checkpoint : {args.checkpoint}')
    print(f'evaluating : {eval_on} / {args.split}'
          + (f'   [CROSS-DATASET: trained on {trained_on}, no fine-tuning]' if cross else ''))

    # `run_eval` expects an options object with these attributes.
    opt = SimpleNamespace(test_batchsize=args.test_batchsize,
                          img_size=cfg['img_size'],
                          color_image=cfg['color_image'])
    data_path = f'{args.data_root}/{eval_on}/'

    tag = f'{eval_on}_{args.split}' + ('_cross' if cross else '')
    save_base = None
    if args.save_predictions:
        save_base = os.path.join(run_dir, 'predictions', tag)
        os.makedirs(save_base, exist_ok=True)

    dice, iou, per_image = run_eval(model, data_path, args.split, opt,
                                    save_base=save_base, device=device)

    # Routing behaviour over the whole split. `run_eval` leaves only the final
    # batch's weights on the routers, so accumulate across a second light pass
    # (inference only, a few seconds) rather than reporting one batch.
    loader = get_loader(image_root=f'{data_path}/{args.split}/images/',
                        gt_root=f'{data_path}/{args.split}/masks/',
                        batchsize=args.test_batchsize, trainsize=cfg['img_size'],
                        shuffle=False, split='test', color_image=cfg['color_image'])
    acc, n_batches = {}, 0
    unc_acc = {}
    with torch.no_grad():
        for images, _, _, _ in loader:
            model(images.to(device))
            for name, s in collect_routing_stats(model).items():
                slot = acc.setdefault(name, {'mean_weight': [0.0] * len(s['mean_weight']),
                                             'select_frac': [0.0] * len(s['select_frac']),
                                             'entropy': 0.0})
                for i, v in enumerate(s['mean_weight']):
                    slot['mean_weight'][i] += v
                for i, v in enumerate(s['select_frac']):
                    slot['select_frac'][i] += v
                slot['entropy'] += s['entropy']
            for k, v in getattr(model, 'last_uncertainty', {}).items():
                unc_acc[k] = unc_acc.get(k, 0.0) + v
            n_batches += 1

    for slot in acc.values():
        slot['mean_weight'] = [v / n_batches for v in slot['mean_weight']]
        slot['select_frac'] = [v / n_batches for v in slot['select_frac']]
        slot['entropy'] /= n_batches
    routing = summarize_routing(acc)
    uncertainty = {k: round(v / n_batches, 4) for k, v in unc_acc.items()} or None

    # --- report ---------------------------------------------------------------
    print(f'\nMean Dice : {dice:.4f}')
    print(f'Mean IoU  : {iou:.4f}')
    if routing:
        print(f'\nRouting over {n_batches} images (kernels {cfg["kernel_sizes"]}, '
              f'{routing["routers"]} routers):')
        print(f'  mean_weight : {routing["mean_weight"]}   (1.0 each = uniform = fixed sum)')
        print(f'  select_frac : {routing["select_frac"]}   (argmax share per kernel)')
        if cfg['routing_mode'].startswith('sparse'):
            # Hard routing weights ARE one-hot, so their entropy is 0 by construction.
            # Reporting that as "collapse" would be wrong -- use select_frac instead.
            print(f'  entropy     : {routing["entropy"]}   (0 by construction for hard '
                  f'routing; read select_frac for collapse)')
            if max(routing['select_frac']) > 0.95:
                print('  NOTE: one kernel takes >95% of selections -- effectively a '
                      'single-kernel model.')
        else:
            print(f'  entropy     : {routing["entropy"]}   (1.0 uniform, 0.0 collapsed)')
            if routing['entropy'] < 0.5:
                print('  NOTE: low entropy -- the router has largely collapsed onto one kernel.')
    if uncertainty:
        print(f'  uncertainty : {uncertainty}   (mean predictive entropy per head)')

    # --- persist ---------------------------------------------------------------
    df = pd.DataFrame(per_image)
    mean_row = df.mean(numeric_only=True).to_dict()
    mean_row['Name'] = 'AVERAGE'
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    xlsx = os.path.join(run_dir, f'per_image_{tag}.xlsx')
    df.to_excel(xlsx, index=False)

    metrics = {'run_id': cfg['run_id'], 'checkpoint': args.checkpoint,
               'routing_mode': cfg['routing_mode'], 'kernel_sizes': cfg['kernel_sizes'],
               'aux_supervision': cfg['aux_supervision'], 'params': n_params,
               'trained_on': trained_on, 'evaluated_on': eval_on, 'split': args.split,
               'cross_dataset': cross, 'n_images': len(per_image),
               'dice': dice, 'iou': iou,
               'sensitivity': mean_row.get('Sensitivity'),
               'specificity': mean_row.get('Specificity'),
               'precision': mean_row.get('Precision'),
               'hd95': mean_row.get('HD95'),
               'routing': routing, 'uncertainty': uncertainty}
    out_json = os.path.join(run_dir, f'test_metrics_{tag}.json')
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\nwrote {out_json}')
    print(f'wrote {xlsx}')
    if save_base:
        print(f'wrote {save_base}/ ({len(per_image)} prediction masks)')


if __name__ == '__main__':
    main()
