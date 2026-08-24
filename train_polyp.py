import os
import time
import random
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project-specific imports
from mkunet_network import MK_UNet
from utils.dataloader_polyp import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter, cal_params_flops


def str2bool(v):
    """argparse type for real boolean flags.

    The original `--augmentation` used a bare `default=True` with no type, so
    `--augmentation False` passed the *string* "False", which is truthy. That made
    it impossible to actually disable augmentation from the command line.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r}')


def resolve_device(pref='auto'):
    """Resolve --device {auto,cpu,cuda} to a torch.device."""
    if pref == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda requested but CUDA is not available')
    if pref == 'auto':
        pref = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(pref)


def set_seed(seed):
    """Seed every RNG the pipeline touches, so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def structure_loss(pred, mask, w=1):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (w * (wbce + wiou)).mean()

def dice_coefficient(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2. * intersection + smooth) / (total + smooth)

def iou(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def test(model, path, dataset, opt, device=None):
    if device is None:
        device = next(model.parameters()).device
    data_path = os.path.join(path, dataset)
    image_root = f'{data_path}/images/'
    gt_root = f'{data_path}/masks/'
    model.eval()
    
    test_loader = get_loader(
        image_root=image_root, gt_root=gt_root, 
        batchsize=opt.test_batchsize, trainsize=opt.img_size,
        shuffle=False, split='test', color_image=opt.color_image,
        num_workers=getattr(opt, 'num_workers', 4)
    )

    DSC, IOU, total_images = 0.0, 0.0, 0
    with torch.no_grad():
        for pack in test_loader:
            images, gts, original_shapes, _ = pack
            images = images.to(device)
            gts = gts.to(device).float()

            ress = model(images)
            # Take the primary output
            predictions = ress[0] if isinstance(ress, list) else ress
            
            for i in range(len(images)):
                # Note: original_shapes in some loaders is [W, H], in others [H, W]
                # We ensure it matches your specific data loader's return order
                h_orig, w_orig = int(original_shapes[0][i]), int(original_shapes[1][i])
                
                # 1. Prediction Resize (Bilinear for soft maps)
                p = predictions[i].unsqueeze(0)
                pred_resized = F.interpolate(p, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                pred_resized = pred_resized.sigmoid().squeeze()
                
                # 2. Local Normalization
                pred_resized = (pred_resized - pred_resized.min()) / (pred_resized.max() - pred_resized.min() + 1e-8)
                
                # 3. GT Resize (NEAREST to maintain binary mask integrity)
                g = gts[i].unsqueeze(0)
                gt_resized = F.interpolate(g, size=(h_orig, w_orig), mode='nearest').squeeze()

                #print(pred_resized.shape, gt_resized.shape, g.shape)

                # 4. Binary Thresholding
                input_binary = (pred_resized >= 0.5).float()
                target_binary = (gt_resized >= 0.2).float() 

                # Applying original thresholding (0.5 for pred, 0.2 for target)
                total_images += 1

                DSC += dice_coefficient(input_binary, target_binary).item()
                IOU += iou(input_binary, target_binary).item()

    return DSC / total_images, IOU / total_images, total_images

def train(train_loader, model, optimizer, epoch, opt, model_name, device):
    model.train()
    global best, test_dice_at_best_val, total_train_time, dict_plot
    
    epoch_start = time.time()
    loss_record = AvgMeter()
    size_rates = [0.75, 1, 1.25] 
    total_step = len(train_loader)

    for i, (images, gts) in enumerate(train_loader, start=1):
        for rate in size_rates:            
            optimizer.zero_grad()
            images, gts = images.to(device), gts.to(device).float()
    
            if rate != 1:
                trainsize = int(round(opt.img_size * rate / 32) * 32)
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='nearest')
            
            out = model(images)
            if opt.deep_supervision and isinstance(out, list) and len(out) > 1:
                # Heads are ordered [p4 (final), p3, p2, p1]; weight them by opt.ds_weights.
                loss = sum(w * structure_loss(p, gts) for w, p in zip(opt.ds_weights, out))
            else:
                loss = structure_loss(out[0] if isinstance(out, list) else out, gts)
            
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                
        if i % 100 == 0 or i == total_step:
            print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {loss_record.show():.4f}')
        
    total_train_time += (time.time() - epoch_start)

    # Save Last
    save_path = opt.train_save
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-last.pth"))

    # Validation and Testing
    epoch_results = {}
    for ds in ['test', 'val']:
        d_dice, d_iou, _ = test(model, opt.test_path, ds, opt, device)
        epoch_results[ds] = d_dice
        logging.info(f'Epoch: {epoch}, Dataset: {ds}, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}')
        print(f'Epoch: {epoch}, Dataset: {ds}, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}')
        dict_plot[ds].append(d_dice)

    # Check if Best Validation Dice
    if epoch_results['val'] > best:
        logging.info(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val']:.4f}) ###")
        print(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val']:.4f}) ###")
        best = epoch_results['val']
        test_dice_at_best_val = epoch_results['test'] # Track test dice at peak val
        torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-best.pth"))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='MK_UNet', 
                        choices=['MK_UNet_T', 'MK_UNet_S', 'MK_UNet', 'MK_UNet_M', 'MK_UNet_L'])    
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005) # base learning rate is 0.0005 for CosineAnnealingLR and 0.0001 for no scheduler
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--test_batchsize', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=300)
    parser.add_argument('--color_image', type=str2bool, default=True)
    parser.add_argument('--augmentation', type=str2bool, default=True)
    parser.add_argument('--dataset', type=str, default='ClinicDB', choices=['ClinicDB', 'ColonDB'])
    parser.add_argument('--data_root', type=str, default='./data/polyp/target')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--runs', type=int, default=5, help='number of independent runs')
    parser.add_argument('--seed', type=int, default=42, help='base seed; run i uses seed+i-1')
    parser.add_argument('--num_workers', type=int, default=4)
    # --- Step 2 extension: supervise the auxiliary heads the baseline discards ---
    parser.add_argument('--deep_supervision', type=str2bool, default=False,
                        help='supervise p1/p2/p3 in addition to the final head (no extra FLOPs)')
    parser.add_argument('--ds_weights', type=float, nargs=4, default=[1.0, 0.5, 0.3, 0.2],
                        metavar=('W_P4', 'W_P3', 'W_P2', 'W_P1'),
                        help='loss weights for [p4 (final), p3, p2, p1]')
    parser.add_argument('--ca_min_squeeze', type=int, default=1,
                        help='floor on the channel-attention squeeze width; 1 = released behaviour')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to a *-resume.pth checkpoint to continue an interrupted run')
    parser.add_argument('--train_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--train_save', type=str, default='')
    opt = parser.parse_args()

    dataset_name = opt.dataset
    if opt.train_path is None:
        opt.train_path = f'{opt.data_root}/{dataset_name}/train/'
    if opt.test_path is None:
        opt.test_path = f'{opt.data_root}/{dataset_name}/'

    device = resolve_device(opt.device)

    # Network configuration mapping
    NET_CONFIGS = {
        'MK_UNet_T': [4, 8, 16, 24, 32],
        'MK_UNet_S': [8, 16, 32, 48, 80],
        'MK_UNet':   [16, 32, 64, 96, 160],
        'MK_UNet_M': [32, 64, 128, 192, 320],
        'MK_UNet_L': [64, 128, 256, 384, 512]
    }

    # Handling Spelling Mistakes or Invalid Choices
    # We use a case-insensitive match or check if it exists in our dictionary
    chosen_net = opt.network
    if chosen_net not in NET_CONFIGS:
        print(f"WARNING: Network '{chosen_net}' not found. Defaulting to 'MK_UNet'.")
        chosen_net = 'MK_UNet'

    # Loaded once; consumed by the first run in the loop so a resumed job continues that run.
    resume_ckpt = torch.load(opt.resume, map_location='cpu') if opt.resume else None

    for run in range(1, opt.runs + 1):
        seed = opt.seed + run - 1
        set_seed(seed)

        dict_plot = {'val': [], 'test': []}
        best = 0.0
        test_dice_at_best_val = 0.0
        total_train_time = 0
        start_epoch = 1

        if resume_ckpt is not None:
            run_id = resume_ckpt['run_id']
        else:
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            run_id = (f"{dataset_name}_{chosen_net}_bs{opt.batchsize}_lr{opt.lr}_"
                          f"e{opt.epoch}_aug{opt.augmentation}_ds{opt.deep_supervision}_"
                          f"ca{opt.ca_min_squeeze}_seed{seed}_run{run}_t{timestamp}")
        opt.train_save = f'./model_pth/{run_id}/'
        
        os.makedirs('logs', exist_ok=True)
        os.makedirs(opt.train_save, exist_ok=True)
        
        logging.basicConfig(filename=f'logs/train_log_{run_id}.log', level=logging.INFO, 
                            format='[%(asctime)s] %(message)s', force=True)

        if opt.network != chosen_net:
            logging.warning(f"User input '{opt.network}' was invalid. Fallback to '{chosen_net}' used.")

        # Build model
        channels = NET_CONFIGS[chosen_net]
        model = MK_UNet(num_classes=1, in_channels=3, channels=channels,
                        deep_supervision=opt.deep_supervision,
                        ca_min_squeeze=opt.ca_min_squeeze)
        model.to(device)

        print(f"Network: {chosen_net} | Channels: {channels} | Device: {device} | Seed: {seed}")
        logging.info(f'run_id={run_id} device={device} seed={seed} dataset={dataset_name} '
                     f'network={chosen_net} channels={channels} img_size={opt.img_size} '
                     f'batchsize={opt.batchsize} lr={opt.lr} epochs={opt.epoch} '
                     f'augmentation={opt.augmentation} deep_supervision={opt.deep_supervision} '
                     f'ds_weights={opt.ds_weights} ca_min_squeeze={opt.ca_min_squeeze} '
                     f'torch={torch.__version__}')
        cal_params_flops(model, opt.img_size, logging, device)
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)

        train_loader = get_loader(
            image_root=f'{opt.train_path}/images/', gt_root=f'{opt.train_path}/masks/',
            batchsize=opt.batchsize, trainsize=opt.img_size,
            shuffle=True, augmentation=opt.augmentation, split='train', color_image=opt.color_image,
            num_workers=opt.num_workers
        )

        if resume_ckpt is not None:
            model.load_state_dict(resume_ckpt['model'])
            optimizer.load_state_dict(resume_ckpt['optimizer'])
            scheduler.load_state_dict(resume_ckpt['scheduler'])
            start_epoch = resume_ckpt['epoch'] + 1
            best = resume_ckpt['best']
            test_dice_at_best_val = resume_ckpt['test_dice_at_best_val']
            total_train_time = resume_ckpt['total_train_time']
            dict_plot = resume_ckpt['dict_plot']
            torch.set_rng_state(resume_ckpt['torch_rng'])
            np.random.set_state(resume_ckpt['numpy_rng'])
            random.setstate(resume_ckpt['python_rng'])
            msg = f'Resumed {run_id} at epoch {start_epoch} (best val Dice so far {best:.4f})'
            print(msg); logging.info(msg)
            resume_ckpt = None  # subsequent runs in --runs start fresh

        for epoch in range(start_epoch, opt.epoch + 1):
            #adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
            train(train_loader, model, optimizer, epoch, opt, run_id, device)
            scheduler.step()
            # Full state snapshot so an OOM/reboot costs one epoch, not the whole run.
            torch.save({
                'run_id': run_id, 'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best': best, 'test_dice_at_best_val': test_dice_at_best_val,
                'total_train_time': total_train_time, 'dict_plot': dict_plot,
                'torch_rng': torch.get_rng_state(),
                'numpy_rng': np.random.get_state(),
                'python_rng': random.getstate(),
            }, os.path.join(opt.train_save, f'{run_id}-resume.pth'))
        # FINAL SUMMARY
        
        summary = (f"\n{'='*40}\nFINAL RESULTS: {run_id}\n"
                   f"Best Val Dice: {best:.4f}\n"
                   f"Test Dice at Best Val: {test_dice_at_best_val:.4f}\n"
                   f"Total Train Time: {total_train_time:.2f}s\n{'='*40}")
        print(summary)
        logging.info(summary)
        
        
