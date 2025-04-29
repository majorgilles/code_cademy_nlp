"""GPT models."""

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
    """Dummy GPT model that just returns the input."""

    def __init__(self, cfg: GPTConfig) -> None:
        """Initialize the GPT model."""
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.embed_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = DummyLayerNorm(cfg.embed_dim)
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GPT model."""
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.size(1)))
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


class DummyTransformerBlock(nn.Module):
    """Dummy transformer block that just returns the input."""

    def __init__(self, cfg: GPTConfig) -> None:
        """Initialize the dummy transformer block."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the dummy transformer block."""
        return x


class DummyLayerNorm(nn.Module):
    """Dummy layer norm that just returns the input."""

    def __init__(self, embed_dim: int) -> None:
        """Initialize the dummy layer norm."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dummy layer norm that just returns the input."""
        return x
