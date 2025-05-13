"""Loss functions for the model.

This module provides functions for calculating the loss during model training.
The main loss function used is cross-entropy loss, which measures how well the model's
predictions match the target values. The loss is calculated by:
1. Flattening the input tensors to match the expected dimensions for cross-entropy
2. Computing the cross-entropy loss between the model's predictions and targets

Note on devices:
A device in PyTorch refers to where the computation is performed - either CPU or GPU.
- CPU (Central Processing Unit): The default device, good for small models and data
- GPU (Graphics Processing Unit): Specialized hardware for parallel processing, much faster for deep learning
- Device is specified as torch.device('cpu') or torch.device('cuda') for GPU
- Moving tensors to the right device is crucial for performance
"""

import torch


def calculate_loss_batch(
    input_batch: torch.Tensor, target_batch: torch.Tensor, model: torch.nn.Module, device: torch.device
) -> torch.Tensor:
    """Calculate the loss for a batch of input and target tensors.

    The function:
    1. Moves input and target tensors to the specified device (CPU or GPU)
    2. Gets model predictions (logits) for the input batch
    3. Flattens the tensors to match cross-entropy requirements:
       - logits: [batch_size, seq_len, vocab_size] -> [batch_size*seq_len, vocab_size]
       - targets: [batch_size, seq_len] -> [batch_size*seq_len]
    4. Computes cross-entropy loss between predictions and targets

    Args:
        input_batch: Input tensor of shape [batch_size, seq_len]
        target_batch: Target tensor of shape [batch_size, seq_len]
        model: The model to generate predictions
        device: Device to move tensors to (e.g., 'cuda' for GPU or 'cpu' for CPU)
               This determines where the computation will be performed

    Returns:
        torch.Tensor: The computed cross-entropy loss
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calculate_loss_loader(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Calculate the loss for a dataloader.

    The function:
    1. Iterates over the dataloader
    2. Calculates the loss for each batch
    3. Returns the average loss
    """
    total_loss = 0.0
    if len(dataloader) == 0:
        return float("nan")
    num_batches = len(dataloader) if num_batches is None else min(num_batches, len(dataloader))

    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
