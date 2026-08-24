"""
Run: python -W ignore scripts/verify_head_mapping.py
"""
import torch

from mkunet_network import MK_UNet

CH = [4, 8, 16, 24, 32]
H = W = 352

captured = {}


def make_hook(name):
    def hook(module, inp, out):
        captured[name] = {'in_res': tuple(inp[0].shape[-2:]),
                          'out_res': tuple(out.shape[-2:]),
                          'in_ch': inp[0].shape[1]}
    return hook


def probe(deep_supervision):
    captured.clear()
    model = MK_UNet(num_classes=1, in_channels=3, channels=CH,
                    deep_supervision=deep_supervision)
    handles = [getattr(model, f'out{i}').register_forward_hook(make_hook(f'out{i}'))
               for i in range(1, 5)]
    model.zero_grad()
    x = torch.randn(2, 3, H, W)
    outs = model(x)
    # Supervise every returned head, as the training loop does.
    loss = sum(o.mean() for o in outs)
    loss.backward()
    for h in handles:
        h.remove()
    grads = {f'out{i}': getattr(model, f'out{i}').weight.grad is not None
             for i in range(1, 5)}
    return outs, grads


print(f'input {H}x{W}, channels {CH}\n')

outs_ds, grads_ds = probe(deep_supervision=True)
outs_no, grads_no = probe(deep_supervision=False)

# The head consuming the finest feature map is the paper's final prediction p1;
# the one consuming the coarsest is the paper's p4. Rank by feature resolution.
order = sorted(captured.items(), key=lambda kv: kv[1]['in_res'][0])  # coarsest first
paper_names = ['p4 (coarsest)', 'p3', 'p2', 'p1 (FINAL)']
mapping = {repo: paper for (repo, _), paper in zip(order, paper_names)}

print(f'{"paper":<14}{"repo var":<10}{"head module":<13}{"feature res":<14}'
      f'{"output res":<13}{"returned (DS off)":<19}{"grad (DS off)":<15}{"grad (DS on)"}')
print('-' * 112)
# Repo returns [p4, p3, p2, p1] when DS on -- i.e. its own naming, final first.
repo_var_of_head = {'out1': 'p1', 'out2': 'p2', 'out3': 'p3', 'out4': 'p4'}
for repo_head, info in order:
    repo_var = repo_var_of_head[repo_head]
    returned_default = 'YES' if repo_var == 'p4' else 'no'
    print(f'{mapping[repo_head]:<14}{repo_var:<10}{repo_head:<13}'
          f'{str(info["in_res"]):<14}{str(info["out_res"]):<13}'
          f'{returned_default:<19}{str(grads_no[repo_head]):<15}{grads_ds[repo_head]}')

print(f'\nreturned heads: DS off -> {len(outs_no)}, DS on -> {len(outs_ds)}')
print(f'all heads emitted at full input resolution: '
      f'{all(tuple(o.shape[-2:]) == (H, W) for o in outs_ds)}')

final_repo = [r for r, p in mapping.items() if p.startswith('p1')][0]
print(f'\n>>> The paper\'s FINAL prediction p1 is the repository\'s '
      f'{repo_var_of_head[final_repo]} (module {final_repo}).')
print('>>> Repository naming is INVERTED relative to the paper.')
print('>>> Never treat the repository\'s p4 as the paper\'s p4.')
