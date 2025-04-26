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
            x: Input tensor of shape [6, 3] where:
               - 6 is the number of tokens
               - 3 is the dimension of each token's embedding
               Example: [[0.43, 0.15, 0.89],  # token 1
                        [0.55, 0.87, 0.66],  # token 2
                        [0.57, 0.85, 0.64],  # token 3
                        [0.22, 0.58, 0.33],  # token 4
                        [0.77, 0.25, 0.10],  # token 5
                        [0.05, 0.80, 0.55]]  # token 6

        Returns:
            Output tensor of shape [6, 2] where:
            - 6 is the number of tokens (same as input)
            - 2 is the output dimension (d_out)
        """
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        return attn_weights @ values
