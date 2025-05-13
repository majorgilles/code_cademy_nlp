"""Dataloader v1."""

import tiktoken
import torch

from src.build_llm_from_scratch_book.dataset import GPTDatasetV1


def create_data_loader_v1(  # noqa: PLR0913
    txt: str,
    batch_size: int = 4,
    context_window_size: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Factory function to create dataloader with dataset.

    Args:
        txt (str): The input text to be tokenized and processed.
        batch_size (int, optional): Number of samples per batch. Defaults to 4.
        context_window_size (int, optional): Size of the context window for token sequences. Defaults to 256.
        stride (int, optional): Number of tokens to skip between consecutive windows. Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to True.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 0.

    Returns:
        torch.utils.data.DataLoader: A PyTorch DataLoader instance containing the processed dataset.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    data_set = GPTDatasetV1(text=txt, tokenizer=tokenizer, context_window_size=context_window_size, stride=stride)

    return torch.utils.data.DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
