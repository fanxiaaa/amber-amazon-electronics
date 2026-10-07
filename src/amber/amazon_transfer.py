"""Detached autoencoder and the shared DMIN history branch."""
import torch
from torch import nn
from amber.amazon_dmin import AmazonDMIN
from amber.dmin_layers import BehaviorRefinerLayer, MultiInterestExtractorLayer


class DetachedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(1024,256),nn.ReLU(),nn.Linear(256,32),nn.Tanh())
        self.decoder=nn.Sequential(nn.Linear(32,256),nn.ReLU(),nn.Linear(256,1024))

    def reconstruction_loss(self, hidden):
        target=hidden.detach()
        return (self.decoder(self.encoder(target))-target).square().mean()


class LoopOnlyVM(AmazonDMIN):
    def add_history_path(self, cfg):
        """Call after recording baseline initialization; preserve its common columns."""
        spec=cfg['loop_encoder']; d=spec['dimension']; heads=spec['heads']
        self.loop_heads=heads
        self.loop_query=nn.Linear(128,d)
        self.loop_refiner=BehaviorRefinerLayer(d,2*d,heads,0,0)
        self.loop_extractor=MultiInterestExtractorLayer(d,2*d,heads,0,0,
            attn_hidden_units=spec['attention_hidden'],attn_activation='ReLU',
            pos_emb_dim=spec['position_dim'],max_seq_len=cfg['cache_history_length'])
        bn,linear=self.dnn.mlp[0],self.dnn.mlp[1]
        old=linear.in_features; new=old+d*heads
        expanded_bn=nn.BatchNorm1d(new,eps=bn.eps,momentum=bn.momentum)
        expanded_linear=nn.Linear(new,linear.out_features)
        with torch.no_grad():
            for key in ('weight','bias','running_mean','running_var'):
                getattr(expanded_bn,key)[:old].copy_(getattr(bn,key))
            expanded_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
            expanded_linear.weight.zero_()
            expanded_linear.weight[:,:old].copy_(linear.weight)
            expanded_linear.bias.copy_(linear.bias)
        self.dnn.mlp[0]=expanded_bn; self.dnn.mlp[1]=expanded_linear

    def forward(self,b):
        target=torch.cat((self.item(b['item']),self.category(b['category'])),-1)
        h=torch.cat((self.item(b['history_item']),self.category(b['history_category'])),-1)
        valid,allowed=self.masks(b['length'],h.shape[1])
        interests=self.extractor(self.refiner(h,allowed),target,allowed,valid)
        z=b['loop_history']; length=b['loop_length']; size=z.shape[1]
        loop_valid=torch.arange(size,device=z.device)[None]<length[:,None]
        eye=torch.eye(size,device=z.device,dtype=torch.bool)
        allowed=(loop_valid[:,None,:]|eye[None])&torch.ones_like(eye).tril()[None]
        allowed=allowed[:,None].expand(-1,self.loop_heads,-1,-1).flatten(0,1)
        extracted=self.loop_extractor(self.loop_refiner(z,allowed),self.loop_query(target),allowed,loop_valid)
        extra=torch.cat(extracted,-1).masked_fill((length==0)[:,None],0)
        return self.dnn(torch.cat(interests+[target,extra],-1)).squeeze(-1)

