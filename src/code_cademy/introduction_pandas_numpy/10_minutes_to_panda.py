"""Series of exercises to get familiar with pandas library."""

# https://pandas.pydata.org/docs/user_guide/10min.html
import numpy as np
import pandas as pd

s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

dates = pd.date_range("20210101", periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
print(df)


df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
)
print(df2)
print(df2.dtypes)

# print head or tail
print(df.head())
print(df.tail(3))

# display index, columns, and values
print(df.index)
print(df.columns)

# convert to numpy array
print(df.to_numpy())

# print types
print(df2.dtypes)
print(df2.to_numpy())

# print summary of data
print(df.describe())

# transpose data (matrix transpose)
print(df.T)

# sort by axis
print(df.sort_index(axis=1, ascending=False))

# sort by values
print(df.sort_values(by="B"))

# get data by label
print(df["A"])
