"""Amazon-specific DMIN adapter; upstream layers/provenance in dmin_layers.py."""
import torch
from torch import nn
from amber.dmin_layers import BehaviorRefinerLayer, MultiInterestExtractorLayer, MLP_Block


class AmazonDMIN(nn.Module):
    def __init__(self, sizes, cfg):
        super().__init__()
        d = cfg['embedding_dim']
        self.heads = cfg['num_heads']
        self.item = nn.Embedding(sizes['item'], d, padding_idx=0)
        self.category = nn.Embedding(sizes['category'], d, padding_idx=0)
        self.refiner = BehaviorRefinerLayer(2*d, 4*d, self.heads,
                                           cfg['attention_dropout'], cfg['net_dropout'])
        self.extractor = MultiInterestExtractorLayer(
            2*d, 4*d, self.heads, cfg['attention_dropout'], cfg['net_dropout'],
            attn_hidden_units=cfg['attention_hidden_units'], attn_activation='Dice',
            pos_emb_dim=cfg['position_dim'], max_seq_len=cfg['history_length'])
        self.dnn = MLP_Block(2*d*(self.heads+1), hidden_units=cfg['dnn_hidden_units'],
                             hidden_activations='ReLU', output_dim=1,
                             dropout_rates=cfg['net_dropout'], batch_norm=True, bn_only_once=True)
        for m in self.modules():
            if type(m) is nn.Linear:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for emb in (self.item, self.category):
            nn.init.normal_(emb.weight, std=1e-4)
            with torch.no_grad():
                emb.weight[0].zero_()
        self.regularizer = cfg['embedding_l2']

    def masks(self, lengths, size):
        valid = torch.arange(size, device=lengths.device)[None, :] < lengths[:, None]
        # Allow padded query diagonal to avoid all-masked softmax; valid queries never read pads.
        eye = torch.eye(size, device=lengths.device, dtype=torch.bool)
        allowed = (valid[:, None, :] | eye[None]) & torch.ones_like(eye).tril()[None]
        return valid, allowed[:, None].expand(-1, self.heads, -1, -1).flatten(0, 1)

    def forward(self, batch):
        target = torch.cat((self.item(batch['item']), self.category(batch['category'])), -1)
        history = torch.cat((self.item(batch['history_item']),
                             self.category(batch['history_category'])), -1)
        valid, allowed = self.masks(batch['length'], history.shape[1])
        refined = self.refiner(history, allowed)
        interests = self.extractor(refined, target, allowed, valid)
        return self.dnn(torch.cat(interests + [target], -1)).squeeze(-1)

    def regularization_loss(self):
        return self.regularizer / 2 * (self.item.weight.square().sum() +
                                      self.category.weight.square().sum())
