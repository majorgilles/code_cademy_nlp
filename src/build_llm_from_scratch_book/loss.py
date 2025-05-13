"""Loss functions for the model.

This module provides functions for calculating the loss during model training.
The main loss function used is cross-entropy loss, which measures how well the model's
predictions match the target values. The loss is calculated by:
1. Flattening the input tensors to match the expected dimensions for cross-entropy
2. Computing the cross-entropy loss between the model's predictions and targets
"""

import torch


def calculate_loss_batch(
    input_batch: torch.Tensor, target_batch: torch.Tensor, model: torch.nn.Module, device: torch.device
) -> torch.Tensor:
    """Calculate the loss for a batch of input and target tensors.

    The function:
    1. Moves input and target tensors to the specified device
    2. Gets model predictions (logits) for the input batch
    3. Flattens the tensors to match cross-entropy requirements:
       - logits: [batch_size, seq_len, vocab_size] -> [batch_size*seq_len, vocab_size]
       - targets: [batch_size, seq_len] -> [batch_size*seq_len]
    4. Computes cross-entropy loss between predictions and targets

    Args:
        input_batch: Input tensor of shape [batch_size, seq_len]
        target_batch: Target tensor of shape [batch_size, seq_len]
        model: The model to generate predictions
        device: Device to move tensors to (e.g., 'cuda' or 'cpu')

    Returns:
        torch.Tensor: The computed cross-entropy loss
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
