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
        """Forward pass.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_in).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_out).
        """
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        return attn_weights @ values
