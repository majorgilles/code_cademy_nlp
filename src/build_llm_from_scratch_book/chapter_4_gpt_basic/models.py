from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GPTConfig:
    """Configuration for the GPT model."""

    vocab_size: int
    embed_dim: int
    context_length: int
    drop_rate: float
    n_layers: int


class DummyGPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.embed_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks == nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = DummyLayerNorm(cfg.embed_dim)
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
