"""Compact self-attention implementation from scratch."""

import torch
from torch import nn


class SelfAttentionV1(nn.Module):
    """Compact self-attention implementation from scratch."""

    def __init__(self, d_in: int, d_out: int) -> None:
        """Initialize the self-attention layer.

        Args:
            d_in (int): The size of the embeddings.
            d_out (int): The size of the output embeddings.
        """
        super().__init__()
        # Initialize the weights for the query, key, and value trainable matrices
        self.W_query = nn.Parameter(torch.randn(d_in, d_out))
        self.W_key = nn.Parameter(torch.randn(d_in, d_out))
        self.W_value = nn.Parameter(torch.randn(d_in, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that computes context vectors using self-attention.

        Args:
            x: Input tensor of e.g. shape [6, 3] where:
               - 6 is the number of tokens
               - 3 is the dimension of each token's embedding
               Example: [[0.43, 0.15, 0.89],  # token 1
                        [0.55, 0.87, 0.66],  # token 2
                        [0.57, 0.85, 0.64],  # token 3
                        [0.22, 0.58, 0.33],  # token 4
                        [0.77, 0.25, 0.10],  # token 5
                        [0.05, 0.80, 0.55]]  # token 6

        Returns:
            Context vectors tensor of shape [6, 2] where:
            - 6 is the number of tokens (same as input)
            - 2 is the output dimension (d_out)
            Each row represents a context vector that combines information from all input tokens,
            weighted by their attention scores relative to that token.
        """
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        return attn_weights @ values


class SelfAttentionV2(nn.Module):
    """Compact self-attention implementation from scratch."""

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
        """Initialize the self-attention layer.

        Args:
            d_in (int): The size of the embeddings.
            d_out (int): The size of the output embeddings.
            qkv_bias (bool): Whether to use query|key|value (QKV) bias in the query, key, and value matrices.
        """
        super().__init__()
        # Initialize the weights for the query, key, and value trainable matrices
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that computes context vectors using self-attention.

        We can improve the SelfAttention_v1 implementation further by utilizing PyTorch’s nn.Linear layers, which
        effectively perform matrix multiplication when the bias units are disabled. Additionally, a significant
        advantage of using nn.Linear instead of manually implementing nn.Parameter(torch.rand(...)) is that nn.Linear
        has an optimized weight initialization scheme, contributing to more stable and effective model training.



        Args:
            x: Input tensor of e.g. shape [6, 3] where:
               - 6 is the number of tokens
               - 3 is the dimension of each token's embedding
               Example: [[0.43, 0.15, 0.89],  # token 1
                        [0.55, 0.87, 0.66],  # token 2
                        [0.57, 0.85, 0.64],  # token 3
                        [0.22, 0.58, 0.33],  # token 4
                        [0.77, 0.25, 0.10],  # token 5
                        [0.05, 0.80, 0.55]]  # token 6

        Returns:
            Context vectors tensor of shape [6, 2] where:
            - 6 is the number of tokens (same as input)
            - 2 is the output dimension (d_out)
            Each row represents a context vector that combines information from all input tokens,
            weighted by their attention scores relative to that token.
        """
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        return attn_weights @ values


class CausalAttention(nn.Module):
    """Causal attention implementation from scratch."""

    def __init__(
        self, d_in: int, d_out: int, context_length: int, dropout_ratio: float = 0.1, qkv_bias: bool = False
    ) -> None:
        """Initialize the causal attention layer.

        We add a  mask to prevent attending to future tokens and a dropout layer to prevent overfitting.

        Args:
            d_in (int): The size of the embeddings.
            d_out (int): The size of the output embeddings.
            context_length (int): The length of the context window.
            dropout_ratio (float): The dropout ratio.
        """
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout_ratio)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that computes context vectors using causal attention.

        Args:
            x: Input tensor of e.g. shape [6, 3] where:
               - 6 is the number of tokens
               - 3 is the dimension of each token's embedding
               Example: [[0.43, 0.15, 0.89],  # token 1
                        [0.55, 0.87, 0.66],  # token 2
                        [0.57, 0.85, 0.64],  # token 3
                        [0.22, 0.58, 0.33],  # token 4
                        [0.77, 0.25, 0.10],  # token 5
                        [0.05, 0.80, 0.55]]  # token 6

        Returns:
            Context vectors tensor of shape [6, 2] where:
        """
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ values
