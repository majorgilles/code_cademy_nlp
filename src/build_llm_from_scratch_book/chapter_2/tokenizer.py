"""Tokenizer class."""

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
        return [self.str_to_int[token] for token in preprocessed]

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of integers into a text."""
        text = " ".join([self.int_to_str[token] for token in tokens])
        text = re.sub(REGEX_PATTERN, r"\1", text)
        logging.info(f"Decoded text: {text}")
        return text
