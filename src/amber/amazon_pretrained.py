"""Public-pretrained AMBER ranking adapter: A/B tokens, alignment then joint fit."""
import torch
from torch import nn
from transformers import AutoModel
from amber.model import AMBERConfig, EventTokenizer


class DefaultRotary(nn.Module):
    """Exact outer product without invoking a JIT batched-matmul specialization."""
    def __init__(self, original):
        super().__init__()
        assert original.rope_type == 'default'
        self.register_buffer('inv_freq', original.inv_freq, persistent=False)
        self.scale = original.attention_scaling

    @torch.no_grad()
    def forward(self, x, position_ids):
        frequency = position_ids.float()[..., None] * self.inv_freq.float()[None, None, :]
        angles = torch.cat((frequency, frequency), -1)
        return (angles.cos() * self.scale).to(x.dtype), (angles.sin() * self.scale).to(x.dtype)


class PretrainedAMBER(nn.Module):
    def __init__(self, sizes, cfg):
        super().__init__()
        torch.manual_seed(cfg['seed'])
        self.tokenizer = EventTokenizer(AMBERConfig(
            categorical_cardinalities=[sizes['item'], sizes['category'], 3],
            categorical_roles=['item', 'item', 'outcome'], d_model=128,
            feature_layers=2, decoder_layers=2, n_heads=4, dropout=0.0))
        self.llm = AutoModel.from_pretrained(cfg['pretrained_path'], local_files_only=True,
            torch_dtype=torch.float32, attn_implementation='sdpa')
        self.llm.rotary_emb = DefaultRotary(self.llm.rotary_emb)
        # All inputs are event embeddings; unused text embeddings need no GPU/optimizer storage.
        self.llm.set_input_embeddings(nn.Identity())
        self.llm.config.use_cache = False
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        d = self.llm.config.hidden_size
        self.projection = nn.Linear(128, d)
        self.head = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.head[-1].bias)

    def set_stage(self, joint):
        self.llm.requires_grad_(joint)

    def event_tokens(self, b):
        n = int(b['length'].max().item())
        batch = len(b['item'])
        cats = torch.zeros(batch, n + 1, 3, dtype=torch.long, device=b['item'].device)
        cats[:, :n, 0] = b['history_item'][:, :n]
        cats[:, :n, 1] = b['history_category'][:, :n]
        cats[:, :n, 2] = 2
        cats[:, n, 0] = b['item']
        cats[:, n, 1] = b['category']
        cats[:, n, 2] = b.get('current_outcome_override', 0)
        nums = torch.empty(batch, n + 1, 0, device=cats.device)
        a = self.projection(self.tokenizer(cats, nums, 'ranking_context'))
        # All provided historical outcomes are reviewed=1. One shared deterministic
        # token is algebraically equivalent and avoids repeatedly encoding the constant.
        label = torch.tensor([[[0, 0, 2]]], dtype=torch.long, device=cats.device)
        z = self.projection(self.tokenizer(label, nums[:1, :1], 'ranking_label'))
        past = torch.stack((a[:, :n], z.expand(batch, n, -1)), dim=2).flatten(1, 2)
        tokens = torch.cat((past, a[:, n:]), dim=1)
        valid = torch.arange(n, device=cats.device)[None] < b['length'][:, None]
        mask = torch.cat((valid.repeat_interleave(2, 1), torch.ones(batch, 1, dtype=torch.bool, device=cats.device)), 1)
        positions = torch.arange(2 * n + 1, device=cats.device)[None].expand(batch, -1).clone()
        positions[:, -1] = 2 * b['length']
        return tokens, mask, positions

    def from_tokens(self, tokens, mask, positions):
        h = self.llm(inputs_embeds=tokens, attention_mask=mask, position_ids=positions,
                     use_cache=False).last_hidden_state[:, -1]
        return self.head(h).squeeze(-1)

    def forward(self, b):
        return self.from_tokens(*self.event_tokens(b))
