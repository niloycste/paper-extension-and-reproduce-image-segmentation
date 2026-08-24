
import ctypes
import platform
import time

import torch
import torch.nn.functional as F

from mkunet_network import MK_UNet


class _MEMSTATUS(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def mem_gb():
    """(available, total) physical memory in GB; (None, None) off Windows."""
    if platform.system() != 'Windows':
        return None, None
    m = _MEMSTATUS()
    m.dwLength = ctypes.sizeof(_MEMSTATUS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30, m.ullTotalPhys / 2 ** 30


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    return (wbce + (1 - (inter + 1) / (union - inter + 1))).mean()


import argparse

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument('--batchsize', type=int, default=8, help='training batch size (default 8, the repo value)')
_ap.add_argument('--img_size', type=int, default=352, help='training resolution (default 352, ClinicDB/ColonDB)')
_ap.add_argument('--epochs', type=int, default=200, help='epoch budget to project (default 200, the paper value)')
_args = _ap.parse_args()

BS, IMG, EPOCHS = _args.batchsize, _args.img_size, _args.epochs
if (BS, IMG) != (8, 352):
    print(f'NOTE: probing at batch {BS} / {IMG}px; projections below are for THAT setting, '
          f'not the standard batch 8 / 352px protocol.\n')
SPLITS = {'ClinicDB': (489, 61, 62), 'ColonDB': (303, 38, 38)}
NET = {'MK_UNet_T': [4, 8, 16, 24, 32], 'MK_UNet': [16, 32, 64, 96, 160]}

avail, total = mem_gb()
print(f'platform : {platform.platform()}')
print(f'cpu      : {platform.processor()}')
print(f'threads  : {torch.get_num_threads()}')
print(f'torch    : {torch.__version__}  (cuda available: {torch.cuda.is_available()})')
if total:
    print(f'memory   : {avail:.1f} GB available / {total:.1f} GB total')
    if avail < 8:
        print('  WARNING: training peaks around 7 GB. Free some memory before starting.')

for name, ch in NET.items():
    model = MK_UNet(num_classes=1, in_channels=3, channels=ch)
    opt = torch.optim.AdamW(model.parameters(), 5e-4, weight_decay=1e-4)

    model.train()
    times = []
    for _ in range(3):
        t0 = time.time()
        for rate in [0.75, 1, 1.25]:                      # multi-scale, as in train_polyp.py
            sz = int(round(IMG * rate / 32) * 32)
            x = torch.randn(BS, 3, sz, sz)
            y = (torch.rand(BS, 1, sz, sz) > 0.7).float()
            opt.zero_grad()
            out = model(x)
            structure_loss(out[0], y).backward()
            opt.step()
        times.append(time.time() - t0)
    step = min(times)

    model.eval()
    inf_times = []
    with torch.no_grad():
        x = torch.randn(BS, 3, IMG, IMG)
        for _ in range(3):
            t0 = time.time()
            model(x)
            inf_times.append(time.time() - t0)
    inf = min(inf_times)

    print(f'\n{name}  (batch {BS}, {IMG}x{IMG})')
    print(f'  multi-scale train step : {step:6.2f} s')
    print(f'  inference batch        : {inf:6.2f} s')
    for ds, (n_tr, n_va, n_te) in SPLITS.items():
        tb = -(-n_tr // BS)
        eb = -(-n_va // BS) + -(-n_te // BS)
        ep = step * tb + inf * eb
        print(f'  {ds:9}: {ep/60:5.2f} min/epoch  ->  {EPOCHS} epochs = {ep*EPOCHS/3600:5.1f} h'
              f'   (4-arm factorial = {4*ep*EPOCHS/3600:5.1f} h)')
