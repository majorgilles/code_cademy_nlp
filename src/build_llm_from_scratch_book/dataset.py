"""Dataset for the GPT model."""

import torch
from tiktoken import Encoding
from torch.utils.data import Dataset


class GPTDatasetV1(Dataset):
    """GPT Dataset V1."""

    def __init__(self, text: str, tokenizer: Encoding, max_length: int, stride: int) -> None:
        """Initialize the dataset with the given text, tokenizer, max_length, and stride."""
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            self.input_ids.append(torch.tensor(input_chunk))

            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the input and target tensors for the given index."""
        return self.input_ids[idx], self.target_ids[idx]
