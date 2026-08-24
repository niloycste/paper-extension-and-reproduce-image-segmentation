
import argparse
import json
import logging
import os
import platform
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reused unchanged from the original training script.
from train_polyp import str2bool, resolve_device, set_seed, structure_loss, test  # noqa: E402
from utils.dataloader_polyp import get_loader  # noqa: E402
from utils.utils import clip_gradient, AvgMeter, cal_params_flops  # noqa: E402
from routing import (build_model, collect_routing_stats,  # noqa: E402
                               summarize_routing, strip_thop_buffers, set_hard_fraction)

EXT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- what to run -------------------------------------------------------
    p.add_argument('--routing_mode', type=str, default='adaptive_soft',
                   choices=['fixed', 'adaptive_soft', 'uncertainty_soft',
                            'sparse_top1', 'sparse_top2'],
                   help="fixed = frozen baseline MK_UNet (F1+F3+F5); adaptive_soft = Stage A; "
                        "uncertainty_soft = Stage B; sparse_top* = Stage C (hard routing, "
                        "unselected branches skipped at inference)")
    p.add_argument('--init_from', type=str, default=None,
                   help='checkpoint to warm-start from (e.g. a trained Stage-A best.pth). '
                        'Parameter names are shared across routing modes, so a soft-routing '
                        'checkpoint initializes a sparse model exactly; missing router keys '
                        'keep their zero-init.')
    p.add_argument('--strict_init', type=str2bool, default=False,
                   help='require the warm-start checkpoint to match every key')
    p.add_argument('--network', type=str, default='MK_UNet_T',
                   choices=['MK_UNet_T', 'MK_UNet_S', 'MK_UNet', 'MK_UNet_M', 'MK_UNet_L'])
    p.add_argument('--kernel_sizes', type=int, nargs='+', default=[1, 3, 5],
                   help='MKDC kernel set; "--kernel_sizes 3" is the single-kernel control')
    p.add_argument('--dataset', type=str, default='ClinicDB', choices=['ClinicDB', 'ColonDB'])

    # --- router ------------------------------------------------------------
    p.add_argument('--router_reduction', type=int, default=4)
    p.add_argument('--router_temperature', type=float, default=1.0,
                   help='>1 flattens the routing distribution, <1 sharpens it')
    p.add_argument('--anneal_epochs', type=int, default=0,
                   help='sparse modes only: linearly ramp soft->hard routing over the '
                        'first N epochs instead of switching abruptly. 0 = switch '
                        'immediately (the default). Inference is always hard.')

    # --- auxiliary (deep) supervision --------------------------------------
    # Stage B needs this: the intermediate heads carry no meaningful uncertainty
    # unless they receive gradient. It is a separate scientific variable, so it
    # must be matched across the arms being compared.
    p.add_argument('--aux_supervision', type=str2bool, default=False,
                   help='supervise the intermediate heads as well as the final one')
    p.add_argument('--aux_loss_weight', type=float, nargs=4, default=[1.0, 0.5, 0.3, 0.2],
                   metavar=('W_FINAL', 'W_3', 'W_2', 'W_COARSE'),
                   help='loss weights, final head first (repo order [p4,p3,p2,p1])')

    # --- optimization (repo defaults, matching the frozen baseline) --------
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--batchsize', type=int, default=8)
    p.add_argument('--test_batchsize', type=int, default=8)
    p.add_argument('--img_size', type=int, default=352)
    p.add_argument('--clip', type=float, default=0.5)
    p.add_argument('--augmentation', type=str2bool, default=True)
    p.add_argument('--color_image', type=str2bool, default=True)
    p.add_argument('--ca_min_squeeze', type=int, default=1)

    # --- bookkeeping -------------------------------------------------------
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--data_root', type=str, default='./data/polyp/target')
    p.add_argument('--out_root', type=str, default=os.path.join(EXT_DIR, 'runs'),
                   help='experiment directory; every run gets its own subfolder')
    p.add_argument('--train_path', type=str, default=None)
    p.add_argument('--test_path', type=str, default=None)
    return p.parse_args()


def train_one_epoch(train_loader, model, optimizer, epoch, opt, device):
    """One epoch of multi-scale training. Returns (mean loss, seconds)."""
    model.train()
    t0 = time.time()
    loss_record = AvgMeter()
    size_rates = [0.75, 1, 1.25]
    total_step = len(train_loader)

    for i, (images, gts) in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images_r, gts_r = images.to(device), gts.to(device).float()
            if rate != 1:
                sz = int(round(opt.img_size * rate / 32) * 32)
                images_r = F.interpolate(images_r, size=(sz, sz), mode='bilinear', align_corners=True)
                gts_r = F.interpolate(gts_r, size=(sz, sz), mode='nearest')

            out = model(images_r)
            if opt.aux_supervision and len(out) > 1:
                loss = sum(w * structure_loss(p, gts_r)
                           for w, p in zip(opt.aux_loss_weight, out))
            else:
                loss = structure_loss(out[0], gts_r)

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        if i % 100 == 0 or i == total_step:
            print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], '
                  f'Step [{i:04d}/{total_step:04d}], '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {loss_record.show():.4f}')
    return float(loss_record.show()), time.time() - t0


def main():
    opt = parse_args()
    if opt.train_path is None:
        opt.train_path = f'{opt.data_root}/{opt.dataset}/train/'
    if opt.test_path is None:
        opt.test_path = f'{opt.data_root}/{opt.dataset}/'
    device = resolve_device(opt.device)

    if opt.routing_mode == 'uncertainty_soft' and not opt.aux_supervision:
        print('WARNING: uncertainty_soft without --aux_supervision True. The intermediate '
              'heads get no gradient, so their uncertainty is meaningless. Enable it, and '
              'compare against an adaptive_soft arm using the SAME setting.')

    for run in range(1, opt.runs + 1):
        seed = opt.seed + run - 1
        set_seed(seed)

        ks = ''.join(str(k) for k in opt.kernel_sizes)
        run_id = (f'{opt.dataset}_{opt.network}_{opt.routing_mode}_k{ks}'
                  f'_aux{opt.aux_supervision}_e{opt.epoch}_seed{seed}'
                  f'_t{time.strftime("%Y%m%d-%H%M%S")}')
        run_dir = os.path.join(opt.out_root, run_id)
        os.makedirs(run_dir, exist_ok=True)

        logging.basicConfig(filename=os.path.join(run_dir, 'train.log'), level=logging.INFO,
                            format='[%(asctime)s] %(message)s', force=True)

        model = build_model(opt.routing_mode, network=opt.network,
                            kernel_sizes=opt.kernel_sizes,
                            deep_supervision=opt.aux_supervision,
                            ca_min_squeeze=opt.ca_min_squeeze,
                            router_reduction=opt.router_reduction,
                            router_temperature=opt.router_temperature).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        if opt.init_from:
            src = strip_thop_buffers(torch.load(opt.init_from, map_location=device))
            res = model.load_state_dict(src, strict=opt.strict_init)
            missing = list(getattr(res, 'missing_keys', []))
            unexpected = list(getattr(res, 'unexpected_keys', []))
            msg = (f'warm-started from {opt.init_from}: '
                   f'{len(src) - len(unexpected)}/{len(model.state_dict())} tensors loaded, '
                   f'{len(missing)} missing, {len(unexpected)} unexpected')
            print(msg)
            logging.info(msg)
            if unexpected:
                # Shared parameter names are the whole point of the warm start, so
                # anything unexpected means the architectures genuinely differ.
                raise SystemExit(f'ERROR: checkpoint has keys the model does not: '
                                 f'{unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')

        cfg = dict(vars(opt))
        cfg.update(run_id=run_id, seed=seed, device=str(device), params=n_params,
                   torch=torch.__version__)
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        with open(os.path.join(run_dir, 'command.txt'), 'w') as f:
            f.write(' '.join([sys.executable] + sys.argv) + '\n')
        # Per-run environment snapshot, so a result can always be traced back to the
        # machine and library versions that produced it.
        with open(os.path.join(run_dir, 'environment.txt'), 'w') as f:
            f.write(f'date        : {time.strftime("%Y-%m-%d %H:%M:%S")}\n'
                    f'platform    : {platform.platform()}\n'
                    f'processor   : {platform.processor()}\n'
                    f'python      : {sys.version.split()[0]}\n'
                    f'torch       : {torch.__version__}\n'
                    f'cuda_avail  : {torch.cuda.is_available()}\n'
                    f'device      : {device}\n'
                    f'torch_threads: {torch.get_num_threads()}\n'
                    f'params      : {n_params}\n')

        print(f'run_id   : {run_id}')
        print(f'routing  : {opt.routing_mode} | kernels {opt.kernel_sizes} | '
              f'aux {opt.aux_supervision} | params {n_params:,} | device {device}')
        logging.info(f'config={json.dumps(cfg)}')
        cal_params_flops(model, opt.img_size, logging, device)
        # thop.profile leaves total_ops/total_params buffers attached to every
        # module; without this they are saved into every checkpoint and the load
        # side is forced to use strict=False, which would also mask real mismatches.
        strip_thop_buffers(model)

        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)
        train_loader = get_loader(
            image_root=f'{opt.train_path}/images/', gt_root=f'{opt.train_path}/masks/',
            batchsize=opt.batchsize, trainsize=opt.img_size, shuffle=True,
            augmentation=opt.augmentation, split='train', color_image=opt.color_image,
            num_workers=opt.num_workers)

        best_val, test_at_best, best_epoch, total_time = 0.0, 0.0, 0, 0.0
        history = []

        for epoch in range(1, opt.epoch + 1):
            # Soft-to-hard annealing (sparse modes only; a no-op otherwise).
            if opt.anneal_epochs > 0:
                alpha = min(1.0, (epoch - 1) / float(opt.anneal_epochs))
                n_set = set_hard_fraction(model, alpha)
                if n_set and (epoch == 1 or alpha in (1.0,) or epoch % 5 == 0):
                    logging.info(f'Epoch: {epoch}, hard_fraction={alpha:.3f}')

            loss, secs = train_one_epoch(train_loader, model, optimizer, epoch, opt, device)
            total_time += secs
            scheduler.step()

            row = {'epoch': epoch, 'loss': loss, 'lr': optimizer.param_groups[0]['lr']}
            for split in ('test', 'val'):
                d, i, _ = test(model, opt.test_path, split, opt, device)
                row[f'{split}_dice'], row[f'{split}_iou'] = d, i
                logging.info(f'Epoch: {epoch}, Dataset: {split}, Dice: {d:.4f}, IoU: {i:.4f}')
                print(f'Epoch: {epoch}, Dataset: {split}, Dice: {d:.4f}, IoU: {i:.4f}')

            summary = summarize_routing(collect_routing_stats(model))
            if summary:
                row['routing'] = summary
                logging.info(f"Epoch: {epoch}, Routing(kernels={opt.kernel_sizes}, "
                             f"{summary['routers']} routers): mean_weight={summary['mean_weight']} "
                             f"select_frac={summary['select_frac']} entropy={summary['entropy']}")
            history.append(row)

            torch.save(model.state_dict(), os.path.join(run_dir, 'last.pth'))
            if row['val_dice'] > best_val:
                logging.info(f"### Best (val Dice {best_val:.4f} -> {row['val_dice']:.4f}) ###")
                print(f"### Best (val Dice {best_val:.4f} -> {row['val_dice']:.4f}) ###")
                best_val, test_at_best, best_epoch = row['val_dice'], row['test_dice'], epoch
                torch.save(model.state_dict(), os.path.join(run_dir, 'best.pth'))

            with open(os.path.join(run_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)

        result = {'run_id': run_id, 'best_val_dice': best_val, 'best_epoch': best_epoch,
                  'test_dice_at_best_val': test_at_best, 'train_time_s': round(total_time, 2),
                  'params': n_params}
        with open(os.path.join(run_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)

        msg = (f"\n{'='*46}\nFINAL: {run_id}\n"
               f"Best Val Dice        : {best_val:.4f} (epoch {best_epoch})\n"
               f"Test Dice @ best val : {test_at_best:.4f}\n"
               f"Train time           : {total_time:.2f}s\n{'='*46}")
        print(msg)
        logging.info(msg)


if __name__ == '__main__':
    main()






"""

  # frozen baseline (should match the Step 1 result)
  python -W ignore "path A/05_train.py" --routing_mode fixed --dataset ClinicDB

  # Stage A -- feature-adaptive
  python -W ignore "path A/05_train.py" --routing_mode adaptive_soft --dataset ClinicDB

  # Stage B -- uncertainty
  python -W ignore "path A/05_train.py" --routing_mode uncertainty_soft \
      --aux_supervision True --dataset ClinicDB

  # single kernel control
  python -W ignore "path A/05_train.py" --routing_mode fixed --kernel_sizes 3
"""
