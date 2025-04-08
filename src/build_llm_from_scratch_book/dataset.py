"""Dataset for the GPT model."""

import torch
from tiktoken import Encoding
from torch.utils.data import Dataset


class GPTDatasetV1(Dataset):
    """Dataset for training a GPT model with sliding window approach."""

    def __init__(self, text: str, tokenizer: Encoding, context_window_size: int, stride: int = 1) -> None:
        """Initialize the dataset with sliding window approach.

        Args:
            text: The text to encode.
            tokenizer: The tokenizer to use.
            context_window_size: The size of the context window (number of tokens the model can see at once).
            stride: Number of tokens to advance the window by each time (controls overlap between sequences).
                  Defaults to 1 for maximum overlap between consecutive sequences.
        """
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - context_window_size, stride):
            input_chunk = token_ids[i : i + context_window_size]
            self.input_ids.append(torch.tensor(input_chunk))

            target_chunk = token_ids[i + 1 : i + context_window_size + 1]
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the input and target tensors for the given index."""
        return self.input_ids[idx], self.target_ids[idx]
