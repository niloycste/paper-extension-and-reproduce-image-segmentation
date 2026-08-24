
import time
import torch
from thop import profile
from mkunet_network import MK_UNet

NET_CONFIGS = {
    'MK_UNet_T': [4, 8, 16, 24, 32],
    'MK_UNet_S': [8, 16, 32, 48, 80],
    'MK_UNet':   [16, 32, 64, 96, 160],
    'MK_UNet_M': [32, 64, 128, 192, 320],
    'MK_UNet_L': [64, 128, 256, 384, 512],
}

# Paper Table 1 reports #Params / #FLOPs at 256x256 input.
PAPER = {
    'MK_UNet_T': (0.027, 0.062),
    'MK_UNet_S': (0.093, 0.125),
    'MK_UNet':   (0.316, 0.314),
    'MK_UNet_M': (1.15,  0.951),
    'MK_UNet_L': (3.76,  3.19),
}

device = torch.device('cpu')
print(f'torch={torch.__version__}  device={device}  cuda_available={torch.cuda.is_available()}\n')
print(f'{"variant":<12}{"params(M)":>11}{"paper":>9}{"FLOPs(G)@256":>14}{"paper":>9}   {"fwd shape @352":>18}  {"bwd":>5}  {"fwd+bwd s":>10}')
print('-' * 100)

for name, ch in NET_CONFIGS.items():
    model = MK_UNet(num_classes=1, in_channels=3, channels=ch).to(device)

    total = sum(p.numel() for p in model.parameters())

    # FLOPs at 256x256 to match the paper's reporting convention.
    x256 = torch.randn(1, 3, 256, 256, device=device)
    flops, _ = profile(model, inputs=(x256,), verbose=False)

    # Forward at the ClinicDB training resolution of 352x352.
    x = torch.randn(2, 3, 352, 352, device=device, requires_grad=False)
    t0 = time.time()
    out = model(x)
    pred = out[0] if isinstance(out, list) else out
    shape_ok = tuple(pred.shape) == (2, 1, 352, 352)

    # Backward.
    loss = pred.mean()
    loss.backward()
    elapsed = time.time() - t0
    grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)

    # Device consistency.
    devs = {p.device.type for p in model.parameters()} | {pred.device.type}
    assert devs == {'cpu'}, f'device mismatch: {devs}'

    p_paper, f_paper = PAPER[name]
    print(f'{name:<12}{total/1e6:>11.4f}{p_paper:>9.3f}{flops/1e9:>14.4f}{f_paper:>9.3f}   '
          f'{str(tuple(pred.shape)):>18}  {"OK" if shape_ok and grads_ok else "FAIL":>5}  {elapsed:>10.2f}')

# Checkpoint save/load round trip on CPU.
print()
m = MK_UNet(num_classes=1, in_channels=3, channels=NET_CONFIGS['MK_UNet_T']).to(device)
torch.save(m.state_dict(), 'ckpt_tmp.pth')
m2 = MK_UNet(num_classes=1, in_channels=3, channels=NET_CONFIGS['MK_UNet_T']).to(device)
m2.load_state_dict(torch.load('ckpt_tmp.pth', map_location=device))
same = all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
print(f'checkpoint save/load on CPU: {"OK" if same else "FAIL"}')
import os
os.remove('ckpt_tmp.pth')
