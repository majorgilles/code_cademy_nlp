"""This module contains functions for generating text with a GPT model."""

import tiktoken
import torch

from src.build_llm_from_scratch_book.modules import GPTModel


def generate_text_simple(model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_size: int) -> torch.Tensor:
    """Generate simple text with untrained model."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    """Convert text to token ids."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    """Convert token ids to text."""
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
