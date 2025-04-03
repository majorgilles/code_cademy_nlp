"""Preprocessing."""

import nltk  # Python library for NLP
import numpy as np

from src.coursera_path.week_1.utils import build_freqs, process_tweet

nltk.download("twitter_samples")

twitter_samples = nltk.corpus.twitter_samples
tweets = twitter_samples.strings("positive_tweets.json")
labels = twitter_samples.strings("negative_tweets.json")

freqs = build_freqs(tweets, labels)
X = np.zeros((len(tweets), 3))


def extract_features(tweet_components: list[str], freqs: dict) -> np.ndarray:
    """Extract features from tweet components."""
    return np.zeros((3,))


m = len(tweets)
for i in range(m):
    p_tweet = process_tweet(tweets[i])
    X[i, :] = extract_features(p_tweet, freqs)
