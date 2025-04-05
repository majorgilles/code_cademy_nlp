"""Chapter 2: Tokenization using NLTK."""

import re

import nltk
from nltk import word_tokenize

# Download required NLTK data
nltk.download("punkt")

# Original example
text = (
    "Trust me, though, the words were on their way, and when "
    "they arrived, Liesel would hold them in her hands like "
    "the clouds, and she would wring them out, like the rain."
)

# New interesting sentences with various linguistic features
new_sentences = [
    "The AI researcher's model achieved 99.9% accuracy - a groundbreaking result!",
    "Mr. Smith bought a Ph.D. degree from example.com for $9,999...",
    "She exclaimed, 'OMG! This can't be real!' while reading the email.",
    "The code runs fast (about 2.5x faster) than our previous implementation.",
]

print("Basic split() tokenization:")
tokens = text.split()
print(tokens[:8])

print("\nRegex tokenization:")
pattern = r"\w+(?:'\w+)?|[^\w\s]"
texts = [text] + new_sentences
tokens = list(re.findall(pattern, texts[-1]))
print(tokens)

print("\nNLTK tokenization:")
for sentence in new_sentences:
    print("\nOriginal:", sentence)
    print("Tokens:", word_tokenize(sentence))
