"""Original companion checks for the pinned JHU-CLSP tutorial.
Usage: python 10_11_jsalt_checks.py /path/to/lab_1_pretraining_a_small_language_model.py
Requires PyTorch (including nn.RMSNorm). Does not install packages or fetch data.
Only the listed imports, constants, functions, and model classes are loaded.
"""
import ast
import math
import sys
from pathlib import Path
import torch
from torch import nn

torch.manual_seed(17)
torch.set_num_threads(1)
source = Path(sys.argv[1]).read_text()
tree = ast.parse(source)
constants = {'MODEL_DIM', 'N_HEADS', 'N_LAYERS', 'DOC_SIZE', 'BATCH_SIZE',
             'EPOCHS', 'VOCAB_SIZE', 'BOS_TOKEN'}
definitions = {'tokenize', 'detokenize', 'build_batches', 'AttnHead', 'Attn',
               'MLP', 'TransformerBlock', 'Transformer'}
selected = []
for node in tree.body:
    if isinstance(node, ast.Import):
        if all(a.name in {'math', 'random', 'torch'} for a in node.names):
            selected.append(node)
        elif all(a.name.startswith('torch.') for a in node.names):
            selected.append(node)
    elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in definitions:
        selected.append(node)
    elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id in constants
                                           for t in node.targets):
        selected.append(node)
namespace = {'__name__': 'jsalt_companion'}
exec(compile(ast.Module(body=selected, type_ignores=[]), '<selected tutorial definitions>', 'exec'), namespace)
assert namespace['tokenize']('é') == [195, 169]
assert namespace['detokenize']([195, 169]) == 'é'
BOS = namespace['BOS_TOKEN']
batch = torch.tensor([[BOS, 97, 98, 99]])
x, y = batch[:, :-1], batch[:, 1:]
assert x.tolist() == [[256, 97, 98]] and y.tolist() == [[97, 98, 99]]
# A small batch reproduces the byte-length issue without the corpus download.
namespace['DOC_SIZE'], namespace['BATCH_SIZE'] = 2, 2
try:
    namespace['build_batches']('aaéabb', torch.device('cpu'))
except (ValueError, RuntimeError):
    print('confirmed: equal character lengths can produce ragged byte-token rows')
else:
    raise AssertionError('expected the pinned character-slicing batch construction to fail')

Attn = namespace['Attn']
loop = Attn(8, 2).eval()
# Fuse the source's independently parameterized heads into three D-wide groups.
class FusedAttention(nn.Module):
    def __init__(self, loop):
        super().__init__()
        self.heads = len(loop.heads)
        self.dh = loop.heads[0].d
        d = self.heads * self.dh
        self.qkv = nn.Linear(d, 3*d)
        self.out = nn.Linear(d, d)
        with torch.no_grad():
            families = ['q_proj', 'k_proj', 'v_proj']
            self.qkv.weight.copy_(torch.cat([
                torch.cat([getattr(h, name).weight for h in loop.heads], dim=0)
                for name in families], dim=0))
            self.qkv.bias.copy_(torch.cat([
                torch.cat([getattr(h, name).bias for h in loop.heads], dim=0)
                for name in families], dim=0))
            self.out.load_state_dict(loop.o_proj.state_dict())

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [u.reshape(b, t, self.heads, self.dh).transpose(1, 2)
                   for u in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        allowed = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        out = (weights @ v).transpose(1, 2).contiguous().reshape(b, t, d)
        return self.out(out)

fused = FusedAttention(loop).eval()
features = torch.randn(2, 4, 8)
with torch.no_grad():
    torch.testing.assert_close(loop(features), fused(features), atol=1e-6, rtol=1e-5)
assert sum(p.numel() for p in loop.parameters()) == 288
assert sum(p.numel() for p in fused.parameters()) == 288
# Validate assembly and sharing, independently of training.
model = namespace['Transformer'](8, 2, 2, max_pos=8).eval()
assert model.blocks[0].attn.heads[0].q_proj.weight is not model.blocks[1].attn.heads[0].q_proj.weight
assert model.embed.weight is not model.lm_head.weight
assert model(x).shape == (1, 3, 257)
# Full-sequence outputs obey the causal dependence boundary.
ids = torch.tensor([[BOS, 97, 98, 99]])
changed = ids.clone()
changed[:, 2:] = torch.tensor([[120, 121]])
with torch.no_grad():
    torch.testing.assert_close(model(ids)[:, :2], model(changed)[:, :2])
# The gated MLP preserves width; it is a different FFN from the vanilla two-map form.
assert model.blocks[0].mlp(features).shape == features.shape
print('byte tokens, shifted pairs, fused-head equivalence, shapes, sharing, and causality checked')
print('attention parameters with biases:', sum(p.numel() for p in loop.parameters()))
