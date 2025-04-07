"""Tokenizing text with tiktoken."""

import tiktoken


def show_tokens(text: str, model: str = "gpt-4") -> None:
    """Show tokens for a given text using a specified model."""
    # Get the tokenizer
    encoder = tiktoken.encoding_for_model(model)

    # Encode the text
    tokens = encoder.encode(text)

    # Show token IDs and their corresponding text
    print(f"\nText: {text}")
    print("\nToken IDs and their corresponding text:")
    for token_id in tokens:
        token_text = encoder.decode([token_id])
        print(f"Token ID: {token_id:4d} | Text: {token_text!r}")


# Example texts
texts = [
    "Hello, how are you?",
    "The quick brown fox jumps over the lazy dog.",
    "GPT-4 is amazing!",
    "I love 🎉 emojis!",
    "The movie 'The Shining' is scary.",
]

# Show tokens for each text
for text in texts:
    show_tokens(text)
    print("\n" + "=" * 50)
