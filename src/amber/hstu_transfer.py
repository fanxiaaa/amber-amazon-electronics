"""Dense HSTU ranking backbone.

HSTU equations follow Zhai et al., ICML 2024, and Meta's Apache-2.0
generative_recommenders/research/modeling/sequential/hstu.py. This independent
dense implementation uses positional bias, SiLU attention, uvqk projection,
gated normalized aggregation and residual output (no softmax attention).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class HSTUBlock(nn.Module):
    def __init__(self, width, heads, max_length):
        super().__init__()
        self.width, self.heads, self.max_length = width, heads, max_length
        self.uvqk = nn.Parameter(torch.empty(width, 4 * width))
        nn.init.normal_(self.uvqk, std=.02)
        self.position_bias = nn.Parameter(torch.empty(2 * max_length - 1))
        nn.init.normal_(self.position_bias, std=.02)
        self.output = nn.Linear(width, width)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, valid, positions):
        b, n, d = x.shape
        u, v, q, k = F.silu(F.layer_norm(x, (d,), eps=1e-6) @ self.uvqk).chunk(4, -1)
        def heads(t): return t.reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        score = heads(q) @ heads(k).transpose(-1, -2)
        relative = positions[:, :, None] - positions[:, None, :]
        score = score + self.position_bias[relative + self.max_length - 1][:, None]
        allowed = (relative >= 0) & valid[:, :, None] & valid[:, None, :]
        # Constant configured N, independent of other examples' history lengths.
        attention = (F.silu(score) / self.max_length).masked_fill(~allowed[:, None], 0)
        value = (attention @ heads(v)).transpose(1, 2).reshape(b, n, d)
        x = x + self.output(u * F.layer_norm(value, (d,), eps=1e-6))
        return x.masked_fill(~valid[..., None], 0)


class HSTURanker(nn.Module):
    def __init__(self, schema, spec, cfg, positive_rate):
        super().__init__()
        self.fields = [i for i, role in enumerate(schema['cat_roles']) if role != 'outcome']
        self.item_fields = spec['history_item_fields']
        self.outcome_index = schema['cat_roles'].index('outcome') if 'outcome' in schema['cat_roles'] else None
        self.visual = 'visual_dim' in schema
        self.categorical = nn.ModuleDict({str(i): nn.Embedding(schema['cat_cardinalities'][i], 64) for i in self.fields})
        num_dim = schema.get('visual_dim', len(schema.get('num_roles', [])))
        self.num_projection = nn.Linear(num_dim, 64) if num_dim else None
        width = cfg['width']
        history_dim = 64 * (len(self.item_fields) + int(self.visual))
        current_dim = 64 * (len(self.fields) + int(num_dim > 0))
        self.history_projection = nn.Linear(history_dim, width)
        self.current_projection = nn.Linear(current_dim, width)
        self.outcome = nn.Embedding(3, width) if not spec['clicked_only'] else None
        self.blocks = nn.ModuleList([HSTUBlock(width, cfg['heads'], spec['history_length'] + 1) for _ in range(cfg['layers'])])
        self.head = nn.Sequential(nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, 1))
        for e in self.categorical.values(): nn.init.normal_(e.weight, std=.02)
        if self.outcome is not None: nn.init.normal_(self.outcome.weight, std=.02)
        nn.init.constant_(self.head[-1].bias, math.log(positive_rate / (1 - positive_rate)))

    def item_features(self, cats, nums):
        parts = [self.categorical[str(i)](cats[..., i]) for i in self.item_fields]
        if self.visual: parts.append(self.num_projection(nums))
        return torch.cat(parts, -1)

    def forward(self, batch):
        b = batch
        h = self.history_projection(self.item_features(b['history_cats'], b['history_nums']))
        if self.outcome is not None:
            h = h + self.outcome(b['history_cats'][..., self.outcome_index])
        current = [self.categorical[str(i)](b['cats'][:, i]) for i in self.fields]
        if self.num_projection is not None: current.append(self.num_projection(b['nums']))
        query = self.current_projection(torch.cat(current, -1))
        n = h.shape[1]
        hist_valid = torch.arange(n, device=h.device)[None] < b['length'][:, None]
        valid = torch.cat((hist_valid, torch.ones_like(hist_valid[:, :1])), 1)
        positions = torch.arange(n + 1, device=h.device)[None].expand(len(h), -1).clone()
        positions[:, -1] = b['length']
        x = torch.cat((h, query[:, None]), 1).masked_fill(~valid[..., None], 0)
        for block in self.blocks: x = block(x, valid, positions)
        return self.head(F.layer_norm(x[:, -1], (x.shape[-1],), eps=1e-6)).squeeze(-1)

    def regularization_loss(self):
        return 0.
