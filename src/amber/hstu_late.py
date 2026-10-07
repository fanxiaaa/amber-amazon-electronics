"""Single-model HSTU with target-conditioned late historical features."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from amber.hstu_transfer import HSTURanker


class LateHistoryHSTU(nn.Module):
    def __init__(self,schema,spec,cfg,positive_rate,raw_control=False):
        super().__init__()
        self.base=HSTURanker(schema,spec,cfg,positive_rate)
        self.raw_control=raw_control
        width=cfg['width'];token_width=width if raw_control else spec['token_dimension']
        self.projection=nn.Linear(token_width,32)
        self.query=nn.Linear(width,32,bias=False)
        self.value=nn.Linear(32,128)
        self.extra_head=nn.Linear(128,256,bias=False)
        nn.init.zeros_(self.extra_head.weight)

    def features(self,b):
        m=self.base
        h=m.history_projection(m.item_features(b['history_cats'],b['history_nums']))
        if m.outcome is not None:h=h+m.outcome(b['history_cats'][...,m.outcome_index])
        current=[m.categorical[str(i)](b['cats'][:,i]) for i in m.fields]
        if m.num_projection is not None:current.append(m.num_projection(b['nums']))
        query=m.current_projection(torch.cat(current,-1));n=h.shape[1]
        valid=torch.arange(n,device=h.device)[None]<b['length'][:,None]
        valid=torch.cat((valid,torch.ones_like(valid[:,:1])),1)
        pos=torch.arange(n+1,device=h.device)[None].expand(len(h),-1).clone();pos[:,-1]=b['length']
        x=torch.cat((h,query[:,None]),1).masked_fill(~valid[...,None],0)
        for block in m.blocks:x=block(x,valid,pos)
        return F.layer_norm(x[:,-1],(x.shape[-1],),eps=1e-6),h

    def forward(self,b):
        q,h=self.features(b);z=h if self.raw_control else b['amber_history'];valid=b['amber_valid']
        z=F.silu(self.projection(F.layer_norm(z,(z.shape[-1],),eps=1e-6)))
        scores=(z*self.query(q)[:,None]).sum(-1)/math.sqrt(32)
        weights=scores.masked_fill(~valid,-1e4).softmax(-1)*valid
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        extra=(self.value(z)*weights[...,None]).sum(1)
        hidden=self.base.head[0](q)+self.extra_head(extra)
        return self.base.head[2](self.base.head[1](hidden)).squeeze(-1)

    def regularization_loss(self):return 0.
