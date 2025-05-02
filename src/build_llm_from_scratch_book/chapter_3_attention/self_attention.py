"""Compact self-attention implementation from scratch."""

import torch
from torch import nn


def _apply_softmax(attn_scores: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Apply scaled softmax to attention scores.

    This private function handles the attention weight computation by:
    1. Scaling the attention scores by 1/√d where d is the head dimension
    2. Applying softmax to get normalized attention weights

    The scaling is crucial because:
    - Dot products in high dimensions grow with √d due to the central limit theorem
    - Without scaling, softmax would produce very sharp distributions (some values ≈1, others ≈0)
    - The scaling factor 1/√d counteracts this effect, keeping attention weights balanced

    Args:
        attn_scores (torch.Tensor): Raw attention scores from query-key dot products
        keys (torch.Tensor): Key vectors used to determine the scaling factor

    Returns:
        torch.Tensor: Normalized attention weights after applying scaled softmax
    """
    return torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)


class SelfAttentionV1(nn.Module):
    """Compact self-attention implementation from scratch.

    This implementation uses a private _apply_softmax method to handle the attention weight computation,
    which includes scaling by the square root of the head dimension to prevent the softmax from producing
    very sharp distributions in high dimensions.
    """

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
        attn_weights = _apply_softmax(attn_scores, keys)
        return attn_weights @ values


class SelfAttentionV2(nn.Module):
    """Compact self-attention implementation from scratch.

    This implementation uses nn.Linear layers instead of manual parameter initialization
    and inherits the _apply_softmax method from SelfAttentionV1 for attention weight computation.
    """

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

        We can improve the SelfAttention_v1 implementation further by utilizing PyTorch's nn.Linear layers, which
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
        attn_weights = _apply_softmax(attn_scores, keys)
        return attn_weights @ values


class CausalAttention(nn.Module):
    """Causal attention implementation from scratch.

    This implementation adds a causal mask to prevent attending to future tokens
    and includes dropout for regularization.
    """

    def __init__(
        self, d_in: int, d_out: int, context_length: int, dropout_ratio: float = 0.1, qkv_bias: bool = False
    ) -> None:
        """Initialize the causal attention layer.

        We add a mask to prevent attending to future tokens and a dropout layer to prevent overfitting.

        Args:
            d_in (int): The size of the embeddings.
            d_out (int): The size of the output embeddings.
            context_length (int): The length of the context window.
            dropout_ratio (float): The dropout ratio.
            qkv_bias (bool): Whether to use query|key|value (QKV) bias in the query, key, and value matrices.
        """
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout_ratio)
        # This is a common pattern when working with PyTorch's register_buffer - we need to explicitly type the buffer
        # to help mypy understand what type it is. The functionality remains exactly the same, we're just helping the
        #   type checker understand our code better.
        self.mask: torch.Tensor  # to make mypy happy and avoid raising warnings
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that computes context vectors using causal attention.

        Args:
            x: Input tensor of e.g. shape [2, 6, 3] where:
               - 2 is the number of batches
               - 6 is the number of tokens
               - 3 is the dimension of each token's embedding
               Example: [[0.43, 0.15, 0.89],  # token 1
                        [0.55, 0.87, 0.66],  # token 2
                        [0.57, 0.85, 0.64],  # token 3
                        [0.22, 0.58, 0.33],  # token 4
                        [0.77, 0.25, 0.10],  # token 5
                        [0.05, 0.80, 0.55]]  # token 6

        Returns:
            Context vectors tensor of shape [2, 6, 2] where:
            - 2 is the number of batches
            - 6 is the number of tokens
            - 2 is the output dimension (d_out)
        """
        # in this case, we have 2 batches of 6 tokens each with 3 dimensions. Num_tokens is 6.
        _, num_tokens, _ = x.shape
        # This works because of how PyTorch's nn.Linear layers handle batched inputs
        # The linear layer will apply the same transformation to each token in the batch independently
        # So, for example, if we have a batch of 2 tokens, the linear layer will apply the same transformation
        # to each token.
        keys = self.W_key(x)  # shape: [2, 6, 2]
        queries = self.W_query(x)  # shape: [2, 6, 2]
        values = self.W_value(x)  # shape: [2, 6, 2]

        # transposes transposes only the inner dimensions (1 = rows aka tokens, 2 = columns aka embeddings)
        # the transpose function is used to change the shape of the tensor (keys) from [2, 6, 2] to [2, 2, 6]
        # so that the dot product of queries and keys is valid
        attn_scores = queries @ keys.transpose(1, 2)
        # First, let's store the sliced mask in a variable with explicit type
        mask_slice: torch.Tensor = self.mask[:num_tokens, :num_tokens]
        # Then convert to boolean and apply the mask
        attn_scores.masked_fill_(mask_slice.bool(), -torch.inf)
        attn_weights = _apply_softmax(attn_scores, keys)
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ values


class MultiHeadAttentionWrapper(nn.Module):
    """Multi-head attention wrapper implementation."""

    def __init__(
        self, d_in: int, d_out: int, context_length: int, dropout_ratio: float, num_heads: int, qkv_bias: bool = False
    ) -> None:
        """Initialize the multi-head attention layer.

        Args:
            d_in (int): The size of the embeddings.
            d_out (int): The size of the output embeddings.
            context_length (int): The length of the context window.
            dropout_ratio (float): The dropout ratio.
            num_heads (int): The number of attention heads.
            qkv_bias (bool): Whether to use query|key|value (QKV) bias in the query, key, and value matrices.
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout_ratio, qkv_bias) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that computes context vectors using multi-head attention."""
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention implementation that processes input sequences in parallel using multiple attention heads.

    This implementation follows the standard transformer architecture where:
    1. Input is split into multiple heads
    2. Each head performs attention independently
    3. Results are concatenated and projected back to the original dimension

    The key difference from MultiHeadAttentionWrapper is that this implementation:
    - Splits the input into heads before attention computation
    - Performs attention in parallel for all heads
    - Uses a final projection layer to combine head outputs
    """

    def __init__(
        self, d_in: int, d_out: int, context_length: int, dropout_ratio: float, num_heads: int, qkv_bias: bool = False
    ) -> None:
        """Initialize the multi-head attention layer.

        Args:
            d_in (int): Input dimension of each token's embedding
            d_out (int): Output dimension of each token's embedding after attention
            context_length (int): Maximum sequence length the model can process
            dropout_ratio (float): Dropout probability for attention weights
            num_heads (int): Number of parallel attention heads
            qkv_bias (bool): Whether to use bias in query/key/value projections
        """
        super().__init__()
        # Ensure output dimension is divisible by number of heads for even splitting
        if (d_out % num_heads) != 0:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Dimension of each head's output

        # Projection (trainable weights) layers for query, key, and value
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Final projection (trainable weights) layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)

        # Regularization
        self.dropout = nn.Dropout(dropout_ratio)

        # Causal mask to prevent attending to future tokens
        self.mask: torch.Tensor  # Type annotation for mypy
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that computes multi-head attention.

        The process involves:
        1. Projecting input into query, key, value spaces
        2. Splitting into multiple heads
        3. Computing attention scores for each head
        4. Applying causal masking and softmax
        5. Computing weighted sum of values
        6. Concatenating head outputs and projecting back

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_tokens, d_in]
                - batch_size: Number of sequences in the batch
                - num_tokens: Length of each sequence
                - d_in: Input embedding dimension

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_tokens, d_out]
                - d_out: Output embedding dimension (same as input if d_out == d_in)
        """
        batch_size, num_tokens, _ = x.shape

        # Project input into query, key, value spaces
        keys = self.W_key(x)  # [batch_size, num_tokens, d_out]
        queries = self.W_query(x)  # [batch_size, num_tokens, d_out]
        values = self.W_value(x)  # [batch_size, num_tokens, d_out]

        # Reshape for multi-head attention
        # Split the embedding dimension into num_heads * head_dim
        keys = keys.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # [batch_size, num_tokens, num_heads, head_dim]
        values = values.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # [batch_size, num_tokens, num_heads, head_dim]
        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # [batch_size, num_tokens, num_heads, head_dim]

        # Transpose to get [batch_size, num_heads, num_tokens, head_dim]
        # This allows parallel computation across heads
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores
        # First transpose keys to [batch_size, num_heads, head_dim, num_tokens]
        transposed_keys = keys.transpose(2, 3)
        # Then compute attention scores [batch_size, num_heads, num_tokens, num_tokens]
        attn_scores = queries @ transposed_keys

        # Apply causal mask to prevent attending to future tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Apply softmax and dropout
        attn_weights = _apply_softmax(attn_scores, keys)
        attn_weights = self.dropout(attn_weights)

        # Compute weighted sum of values
        # [batch_size, num_heads, num_tokens, head_dim]
        context_vectors = attn_weights @ values

        # First transpose to get heads and dimensions together
        # [batch_size, num_tokens, num_heads, head_dim]
        context_vectors = context_vectors.transpose(1, 2)
        # Then reshape to combine head outputs into final dimension
        # [batch_size, num_tokens, d_out] where d_out = num_heads * head_dim
        context_vectors = context_vectors.contiguous().view(batch_size, num_tokens, self.d_out)

        # Final projection
        return self.out_proj(context_vectors)


if __name__ == "__main__":
    torch.manual_seed(123)
    batch_size, context_length, d_in, d_out, num_heads = 2, 6, 3, 2, 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    batch = torch.randn(batch_size, context_length, d_in)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)


"""
# Raw tokens
[
    [
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9, 10, 11, 12]
    ],
    [
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]
    ]
]

# W matrices
[
    [
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9, 10, 11, 12]
    ],
    [
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]
    ]
]

# After view
[
    [
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]]
        ],
        [
            [[13, 14], [15, 16]],
            [[17, 18], [19, 20]],
            [[21, 22], [23, 24]]
        ]
    ]
]

# After transpose(1, 2)
[
    [
        [
            [[1, 2], [3, 4]],
            [[13, 14], [15, 16]]
        ],
        [
            [[5, 6], [7, 8]],
            [[17, 18], [19, 20]]
        ],
        [
            [[9, 10], [11, 12]],
            [[21, 22], [23, 24]]
        ]
    ]
]
"""
