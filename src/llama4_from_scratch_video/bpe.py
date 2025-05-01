"""Taken from https://www.youtube.com/watch?v=biveB0gOlak&t=1s.

Create a BPE from scratch.
"""

import collections

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


def get_pair_stats(splits: dict) -> dict:
    """Get the pair counts for all pairs in the splits."""
    pair_count: dict[tuple[str, str], int] = collections.defaultdict(int)
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_count[pair] += freq
    return pair_count


print("Pair counts:", get_pair_stats(word_splits))


def merge_pair(pair_to_merge: tuple[str, str], splits: dict) -> dict:
    """Merges a pair of adjacent symbols in the vocabulary splits.

    This function is a key part of the Byte Pair Encoding (BPE) algorithm. It takes a pair of symbols
    that should be merged and updates all occurrences of this pair in the vocabulary splits by
    replacing them with a single merged token.

    Args:
        pair_to_merge: A tuple containing two adjacent symbols to be merged (e.g., ('T', 'h')).
        splits: A dictionary mapping word tuples to their frequencies in the corpus.

    Returns:
        A new dictionary with the same structure as splits, but where all occurrences of the specified
        pair have been replaced with a single merged token.

    Example:
        If pair_to_merge is ('T', 'h') and splits contains ('T', 'h', 'i', 's', '</w>'),
        the result will contain ('Th', 'i', 's', '</w>') with the same frequency.
    """
    new_splits = {}
    (first, second) = pair_to_merge
    merged_token = first + second
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        new_symbols = []
        i = 0
        num_symbols = len(symbols)
        while i < num_symbols:
            if i < num_symbols - 1 and symbols[i] == first and symbols[i + 1] == second:
                new_symbols.append(merged_token)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_splits[tuple(new_symbols)] = freq
    return new_splits


print("Merged splits:", merge_pair(("T", "h"), word_splits))
print("Merged splits:", merge_pair(("i", "s"), word_splits))

def iterate_algo(current_splits: dict, num_merges: int, vocabulary: list[str]) -> tuple[dict, dict, list[str]]:
    """Iterate the BPE algorithm."""
    merges = {}
    print("\n--- Starting BPT merges ---")
    print(f"Initial Splits: {current_splits}")
    print("-" * 30)

    for i in range(num_merges):
        print(f"\nMerge iteratation {i+1}/{num_merges}")

        pair_stats = get_pair_stats(current_splits)
        if not pair_stats:
            print("No more pairs to merge.")
            break

        sorted_pairs = sorted(pair_stats.items(), key = lambda item: item[1], reverse=True)
        print(f"Top 5 pair frequencies: {sorted_pairs[:5]}")

        best_pair = max(pair_stats, key=pair_stats.get)  # type: ignore
        best_freq = pair_stats[best_pair]
        print(f"Found best pair: {best_pair} with frequency {best_freq}")

        current_splits = merge_pair(best_pair, current_splits)
        new_token = best_pair[0] + best_pair[1]
        print(f"Merging {best_pair} into {new_token}")
        print(f"Splits after merge: {current_splits}")

        vocab.append(new_token)
        print(f"Updated vocab: {vocab}")

        merges[best_pair] = new_token
        print(f"Updated merges: {merges}")

        print("-" * 30)
    return merges, current_splits, vocab

num_merges = 15
merges, current_splits, vocab = iterate_algo(word_splits, num_merges=num_merges, vocabulary=vocab)

print("\n--- Final merges ---")
for pair_token, merged_token in merges.items():
    print(f"{pair_token} -> {merged_token}")

print("\n--- Final splits ---")
print(current_splits)

print("\n--- Final vocabulary ---")
print(vocab)
