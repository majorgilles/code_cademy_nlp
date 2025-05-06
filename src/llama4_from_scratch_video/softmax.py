"""A simple softmax implementation."""

import torch

# config
hidden_size = 128  # Diemnsionality of the embedding vector
num_attention_heads = 16  # Number of attention heads
num_key_value_heads = 4  # Number of key value heads
head_dim = hidden_size // num_attention_heads  # Dimension of each head
max_position_embeddings = 256  # Maximum number of position embeddings
rope_theta = 10_000.0  # Base for the RoPE frequency calculation
rms_norms_eps = 1e-5  # Epsilon for RMSNorm
attention_bias = False  # Whether to use attention bias in Q K V O projections
attention_dropout = 0.0  # Dropout probabilityfor attention weights
use_qk_norm = True  # Whether to appluy L2 norm to Q and K before attention

# sample input
batch_size = 2
sequence_length = 10
hidden_states = torch.randn(batch_size, sequence_length, hidden_size)
# Shape: (batch_size, sequence_length, hidden_size)
position_ids = torch.arange(0, sequence_length).unsqueeze(0).repeat(batch_size, 1)
