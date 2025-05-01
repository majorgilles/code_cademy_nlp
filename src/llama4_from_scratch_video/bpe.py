"""Taken from https://www.youtube.com/watch?v=biveB0gOlak&t=1s.

Create a BPE from scratch.
"""

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
end_of_word = "</w>"
unique_chars = set(end_of_word)

for doc in corpus:
    for char in doc:
        unique_chars.add(char)

vocab = list(unique_chars)
vocab.sort()

print("Initial vocab:", vocab)
print("Initial vocab length:", len(vocab))


word_splits = {}
for doc in corpus:
    words = doc.split(" ")
    for word in words:
        if word:
            word_tuple = tuple(list(word) + [end_of_word])
            if word_tuple not in word_splits:
                word_splits[word_tuple] = 0
            word_splits[word_tuple] += 1

print("Word splits:", word_splits)
