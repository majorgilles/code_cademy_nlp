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


def _apply_softmax(attn_scores: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Apply scaled softmax to attention scores for self-attention.

    This function computes attention weights by:
    1. Scaling raw query-key dot products by 1/√d, where d is the dimensionality of the key vectors.
    2. Applying softmax to convert the scores into a probability distribution.

    The full formula for context vector calculation is:

    Attention(Q, K, V) = softmax((Q @ K.T) / sqrt(d)) @ V

    So we compute the attention weights by:
    attn_scores = queries @ keys.T
    attn_weights = _apply_softmax(attn_scores, keys)
    Then we compute the context vector by:
    context_vector = attn_weights @ values

    Scaling explanation:

    - **Scaling for numerical stability**:
      In high-dimensional spaces, dot products can become large in magnitude, which causes
      the softmax output to be overly sharp (i.e., nearly one-hot).
      Scaling by √d (the square root of the dimensionality of the key vectors)  mitigates this
       effect, producing more balanced and useful gradients during training.

    - **Softmax for interpretability**:
      Raw dot products don't naturally form a probability distribution. Applying softmax
      ensures that attention weights are non-negative and sum to 1, making them interpretable
      as the relative importance of each token.

    - **Effective value weighting**:
      These normalized weights are then used to compute a weighted sum over the value vectors,
      forming the basis of the attention output.

    Args:
        attn_scores (torch.Tensor): Raw attention scores from query-key dot products.
        keys (torch.Tensor): Key vectors used to determine the scaling factor (dimensionality).

    Returns:
        torch.Tensor: Normalized attention weights.
    """
    scale = keys.shape[-1] ** 0.5
    return torch.softmax(attn_scores / scale, dim=-1)


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

        The non approximate version is:
        gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        (erf is the error function, which is the integral of the Gaussian distribution)

        The approximate version is:
        gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

        Why the approximate version?
        - The exact version is computationally expensive to compute.
        - The approximate version is a good approximation of the exact version.
        - The approximate version is faster to compute.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
        """
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    """Feed-Forward network that applies a linear transformation followed by a non-linear activation function.

    This module consists of two linear transformations with a GELU activation function in between.
    The first linear transformation projects the input to a higher-dimensional space,
    while the second linear transformation projects it back to the original dimension.
    The GELU activation function introduces non-linearity between the two linear transformations.

    The first linear layer expands the embedding dimension by a factor of 4 (4 * embed_dim).
    This expansion allows the network to learn more complex patterns by providing a larger
    intermediate representation space. The second linear layer then compresses this expanded
    representation back to the original embedding dimension. This "bottleneck" architecture
    (expand -> process -> compress) is a common pattern in transformer models that helps
    capture more complex relationships while maintaining computational efficiency.

    Note on tensor dimensions:
    The Linear layer only operates on the last dimension of the input tensor. For example,
    if the input is of shape (batch_size, seq_len, embed_dim), the Linear layer will:
    1. Keep the first two dimensions (batch_size, seq_len) unchanged
    2. Only transform the last dimension (embed_dim) to (4 * embed_dim)
    This means the output shape will be (batch_size, seq_len, 4 * embed_dim)
    The same applies to the second Linear layer which compresses back to (batch_size, seq_len, embed_dim)

    Args:
        embed_dim (int): The dimension of the input features
        hidden_dim (int): The dimension of the hidden features
    """

    def __init__(self, cfg: GPTConfig) -> None:
        """Initialize the FeedForward module.

        Args:
            cfg (GPTConfig): The configuration for the FeedForward module
        """
        super().__init__()
        self.layers = nn.Sequential(
            # Expand the embedding dimension by 4x to allow for more complex pattern learning
            # Input shape: (batch_size, seq_len, embed_dim)
            # Output shape: (batch_size, seq_len, 4 * embed_dim)
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            GELU(),
            # Compress back to the original embedding dimension
            # Input shape: (batch_size, seq_len, 4 * embed_dim)
            # Output shape: (batch_size, seq_len, embed_dim)
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the FeedForward module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
        """
        return self.layers(x)


class ExampleDeepNeuralNetwork(nn.Module):
    """Example deep neural network that demonstrates the use of linear layers and activation functions.

    This class provides a simple example of a deep neural network using PyTorch. It consists of
    two linear layers with a GELU activation function in between. The input tensor is first projected
    to a higher-dimensional space, then the GELU activation function introduces non-linearity,
    and finally the output is compressed back to the original dimension.

    Args:
        input_dim (int): The dimension of the input features
        hidden_dim (int): The dimension of the hidden features
    """

    def __init__(self, layer_sizes: list[int], use_shortcut: bool) -> None:
        """Initialize the ExampleDeepNeuralNetwork.

        Args:
            layer_sizes (list[int]): The dimensions of the layers
            use_shortcut (bool): Whether to use a shortcut connection
        """
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ExampleDeepNeuralNetwork.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
        """
        for layer in self.layers:
            layer_output = layer(x)
            x = x + layer_output if self.use_shortcut and x.shape == layer_output.shape else layer_output
        return x


class TransformerBlock(nn.Module):
    """Transformer block that implements a single layer of the transformer architecture.

    The transformer block consists of two main components:
    1. Multi-head self-attention mechanism (self.att)
    2. Feed-forward neural network (self.ff)

    Each component is wrapped in a shortcut connection (also called skip connection or residual connection)
    and layer normalization, following the "Pre-LN" (Pre-Layer Normalization) architecture. This architecture
    has been shown to provide more stable training compared to the original "Post-LN" architecture.

    Shortcut Connections Explained:
    - A shortcut connection allows information to flow directly from one layer to another by
      adding the input to the output of a transformation
    - In this implementation, we have two shortcut connections:
      1. First shortcut: x = x + Dropout(Attention(LayerNorm(x)))
         - The original input (stored in 'shortcut') is added to the transformed output
         - This helps with gradient flow during training
      2. Second shortcut: x = x + Dropout(FeedForward(LayerNorm(x)))
         - Similar to the first shortcut, but after the feed-forward network
    - The shortcut connections are implemented using the '+' operator in the forward pass
    - They help prevent the vanishing gradient problem and make it easier to train deep networks

    Args:
        cfg (GPTConfig): Configuration object containing model hyperparameters
    """

    def __init__(self, cfg: GPTConfig) -> None:
        """Initialize the TransformerBlock.

        Args:
            cfg (GPTConfig): The configuration for the TransformerBlock containing:
                - embed_dim: Dimension of the input embeddings
                - context_length: Maximum sequence length
                - n_heads: Number of attention heads
                - qkv_bias: Whether to use bias in QKV projections
                - drop_rate: Dropout probability for regularization
        """
        super().__init__()
        # Multi-head attention block
        self.att = MultiHeadAttention(
            d_in=cfg.embed_dim,
            d_out=cfg.embed_dim,
            context_length=cfg.context_length,
            dropout_ratio=cfg.drop_rate,
            num_heads=cfg.n_heads,
            qkv_bias=cfg.qkv_bias,
        )
        # Feed-forward network block
        self.ff = FeedForward(cfg)
        # Layer normalization layers
        self.norm1 = LayerNorm(cfg.embed_dim)
        self.norm2 = LayerNorm(cfg.embed_dim)
        # Dropout layer that is applied before the shortcut connection
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TransformerBlock.

        The forward pass implements the following computation:
        1. x_norm = LayerNorm(x)
        2. att_out = MultiHeadAttention(x_norm)
        3. x = x + Dropout(att_out)  # First shortcut connection
        4. x_norm = LayerNorm(x)
        5. ff_out = FeedForward(x_norm)
        6. x = x + Dropout(ff_out)   # Second shortcut connection

        The drop_shortcut layer (self.drop_shortcut) Explained:
        - This is a dropout layer that is applied to the output of each sub-block (attention and feed-forward)
          BEFORE it is added to the shortcut connection
        - The name 'drop_shortcut' comes from its position in the architecture: it drops (randomly zeros)
          some elements of the transformed output before it takes the "shortcut" path to be added
          to the original input
        - For example, in the attention block:
          1. x is transformed by attention: att_out = Attention(LayerNorm(x))
          2. drop_shortcut randomly zeros some elements: dropped = Dropout(att_out)
          3. The dropped output is added to the original input: x = x + dropped
        - This specific placement of dropout (before the shortcut addition) is crucial because:
          - It only affects the transformed features, not the original input
          - The original input remains intact through the shortcut path
          - This creates a form of regularization that forces the network to learn robust features
            while maintaining the benefits of the shortcut connection
        - The dropout probability is controlled by cfg.drop_rate

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor of the same shape as input
        """
        # First sub-block: Multi-head attention with shortcut connection
        shortcut = x  # Store input for shortcut connection
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)  # Apply dropout to transformed output before shortcut
        x = x + shortcut  # Shortcut connection: add original input to transformed output

        # Second sub-block: Feed-forward network with shortcut connection
        shortcut = x  # Store input for shortcut connection
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)  # Apply dropout to transformed output before shortcut
        return x + shortcut  # Shortcut connection: add original input to transformed output


class GPTModel(nn.Module):
    """GPT model that implements the transformer architecture.

    The GPT model consists of a stack of transformer blocks, each containing:
    - Multi-head self-attention mechanism


    Args:
        cfg (GPTConfig): Configuration object containing model hyperparameters
    """

    def __init__(self, cfg: GPTConfig) -> None:
        """Initialize the GPTModel.

        Args:
            cfg (GPTConfig): The configuration for the GPTModel containing:
                - embed_dim: Dimension of the input embeddings
                - context_length: Maximum sequence length
                - n_heads: Number of attention heads
                - qkv_bias: Whether to use bias in QKV projections
                - drop_rate: Dropout probability for regularization
        """
        super().__init__()
        self.token_embeddings = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.positional_embeddings = nn.Embedding(cfg.context_length, cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.embed_dim)
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GPTModel.

        Args:
            in_idx (torch.Tensor): Input tensor of shape (batch_size, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        _, sequence_length = in_idx.shape
        token_embeddings = self.token_embeddings(in_idx)
        positional_embeddings = self.positional_embeddings(torch.arange(sequence_length, device=in_idx.device))
        x = token_embeddings + positional_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
