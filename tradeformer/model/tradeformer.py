import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class TradeFormerConfig:
    block_size: int = 256
    vocab_size: int = 1000
    n_layers: int = 6
    n_heads: int = 8
    n_embed: int = 512


class CausalSelfAttention(nn.Module):

    def __init__(self, cfg: TradeFormerConfig):
        super().__init__()

        assert cfg.n_embed % cfg.n_heads == 0

        self.c_attn = nn.Linear(cfg.n_embed, 3 * cfg.n_embed)
        self.c_proj = nn.Linear(cfg.n_embed, cfg.n_embed)

        self.n_head = cfg.n_heads
        self.n_embed = cfg.n_embed

        self.register_buffer('bias', torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                             .view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):

        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)

        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # B, nh, T, hs
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # B, nh, T, hs
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # B, nh, T, hs

        att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, cfg: TradeFormerConfig):
        super().__init__()

        self.c_fc = nn.Linear(cfg.n_embed, 4 * cfg.n_embed)
        self.gelu = nn.SiLU()
        self.c_proj = nn.Linear(4 * cfg.n_embed, cfg.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, cfg: TradeFormerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embed)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embed)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TradeFormer(nn.Module):

    def __init__(self, cfg: TradeFormerConfig):
        super(TradeFormer, self).__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embed),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embed),
                h=nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers)),
                ln_f=nn.LayerNorm(cfg.n_embed)
            )
        )

        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()

        assert T <= self.cfg.block_size, f"Cannot forward sequence length of {
            T}, block size {self.cfg.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return (logits, loss)
