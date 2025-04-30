"""GPT models."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GPTConfig:
    """Configuration for the GPT model."""

    vocab_size: int  # tokenizer vocab size
    embed_dim: int  # embedding dimension
    context_length: int  # context length
    drop_rate: float  # dropout rate
    n_layers: int  # number of layers
    n_heads: int  # number of attention heads
    qkv_bias: bool  # whether to use bias in the qkv layer


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

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GPT model."""
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
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


class LayerNorm(nn.Module):
    """Layer Normalization module that normalizes input features across the last dimension.

    Layer Normalization is a technique used to normalize the activations of a neural network
    layer. It helps stabilize training by ensuring that the inputs to each layer have consistent
    statistics (mean and variance) across different examples in a batch. This is important because
    as the network trains, the distribution of inputs to each layer can change significantly
    (a problem known as internal covariate shift), making training unstable. Layer normalization
    counteracts this by normalizing the inputs to have zero mean and unit variance (1).

    This implementation follows the standard layer normalization formula:

    y = scale * (x - mean) / sqrt(var + eps) + shift

    where:
    - x is the input tensor
    - mean and var are computed across the last dimension
    - scale and shift are learnable parameters
    - eps is a small constant for numerical stability

    Attributes:
        eps (float): Small constant added to variance for numerical stability
        scale (nn.Parameter): Learnable scaling parameter of shape (embed_dim,)
        shift (nn.Parameter): Learnable shift parameter of shape (embed_dim,)
    """

    def __init__(self, embed_dim: int) -> None:
        """Initialize the Layer Normalization module.

        Args:
            embed_dim (int): The dimension of the input features to be normalized
        """
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Gaussian Error Linear Unit (GELU) activation function.

    The GELU activation function is defined as:

    gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

    where erf is the error function.

    Attributes:
        scale (float): Scaling factor for the activation function
    """

    def __init__(self, scale: float = 1.0) -> None:
        """Initialize the GELU activation function.

        Args:
            scale (float): Scaling factor for the activation function
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GELU activation function.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
        """
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
