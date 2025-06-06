"""Tokenizer classes."""

import logging
import re

REGEX_PATTERN = r'([,.?_!"()\']|--|\s)'


class SimpleTokenizerV1:
    """Simple tokenizer that splits text into tokens based on whitespace and punctuation."""

    def __init__(self, vocabulary: dict[str, int]) -> None:
        """Initialize the tokenizer with a vocabulary."""
        self.str_to_int = vocabulary
        self.int_to_str = {v: k for k, v in self.str_to_int.items()}

    def encode(self, text: str) -> list[int]:
        """Encode a text into a list of integers."""
        preprocessed = re.split(REGEX_PATTERN, text)
        preprocessed = [token for token in preprocessed if token and token.strip()]
        result = [self.str_to_int[token] for token in preprocessed]
        logging.info(f"Encoded text: {result}")
        return result

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of integers into a text."""
        text = " ".join([self.int_to_str[token] for token in tokens])
        text = re.sub(r"\s+([,.?!\"()\'])", r"\1", text)  # this is to remove extra spaces before punctuation
        logging.info(f"Decoded text: {text}")
        return text


class SimpleTokenizerV2:
    """Simple tokenizer that splits text into tokens based on whitespace and punctuation.

    Differs from SimpleTokenizerV1 in that it handles unknown tokens and adds special tokens.
    """

    def __init__(self, vocab: dict[str, int]) -> None:
        """Initialize the tokenizer with a vocabulary."""
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Encode a text into a list of integers."""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids: list[int]) -> str:
        """Decode a list of integers into a text."""
        text = " ".join([self.int_to_str[i] for i in ids])

        return re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
