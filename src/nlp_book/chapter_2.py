"""Chapter 2: Tokenization."""

import re

text = (
    "Trust me, though, the words were on their way, and when "
    "they arrived, Liesel would hold them in her hands like "
    "the clouds, and she would wring them out, like the rain."
)
tokens = text.split()
print(tokens[:8])

pattern = r"\w+(?:\'\w+)?|[^\w\s]"
texts = [text]
texts.append("There's no such thing as survival of the fittest. " "Survival of the most adequate, maybe.")
tokens = list(re.findall(pattern, texts[-1]))
print(tokens[:8])
print(tokens[8:16])
print(tokens[16:])
