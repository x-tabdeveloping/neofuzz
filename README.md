# neofuzz

Blazing fast fuzzy text search for Python.

## Introduction

neofuzz is a fuzzy search library based on vectorization and approximate nearest neighbour
search techniques.

What neofuzz is good at:
  - Hella Fast.
  - Repeated searches in the same space of options.
  - Compatibility with already existing TheFuzz code.
  - Incredible flexibility in the vectorization process.
  - Complete control over the nearest neighbour algorithm's speed and accuracy.

If you're looking for a scalable solution for searching the same, large dataset
with lower quality of results but incredible speed, neofuzz is the thing you're looking for.

What neofuzz is not good at:
  - Exact and certainly correct results.
  - Searching different databases in succession.
  - Not the best fuzzy search algorithm.

If you're looking for a library that's great for fuzzy searching small amount of data with a
good fuzzy algorithm like levenshtein or hamming distance, neofuzz is probably not
the thing for you.

## Usage

The base abstraction of neofuzz is the `Process`, which is a class aimed at replicating TheFuzz's API.

A `Process` takes a vectorizer, that turns strings into vectorized form, and different parameters
for fine-tuning the indexing process.

If you want a plug-and play experience you can create a generally good quick and dirty
process with the `char_ngram_process()` process.

```python
from neofuzz import char_ngram_process

# We create a process that takes character 1 to 5-grams as features for
# vectorization and uses a tf-idf weighting scheme.
# We will use cosine distance for the nearest neighbour search.
process = char_ngram_process(ngram_range=(1,5), metrics="cosine", tf_idf=True)

# We index the options that we are going to search in
process.index(options)

# Then we can extract the ten most similar items the same way as in
# thefuzz
process.extract("fuzz", limit=10)
---------------------------------
[('fuzzer', 67),
 ('Januzzi', 30),
 ('Figliuzzi', 25),
 ('Fun', 20),
 ('Erika_Petruzzi', 20),
 ('zu', 20),
 ('Zo', 18),
 ('blog_BuzzMachine', 18),
 ('LW_Todd_Bertuzzi', 18),
 ('OFU', 17)]
```

If you want to use a custom vectorization process with dimentionality reduction for example,
you are more than free to do so by creating your own custom `Process`

```python
from neofuzz import Process

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

# Let's say we have a list of sentences instead of words,
# Then we can use token ngrams as features
tf_idf = TfidfVectorizer(analyzer="word", stop_words="english", ngram_range=(1,3))
# We use NMF for reducing the dimensionality of the vectors to 20
# This could improve speed but takes more time to set up the index
nmf = NMF(n_components=20)

# Our vectorizer is going to be a pipeline
vectorizer = make_pipeline(tf_idf, nmf)

# We create a process and index it with our corpus.
process = Process(vectorizer, metric="cosine")
process.index(sentences)

# Then you can extract results
process.extract("she ate the cat", limit=3)
-------------------------------------------
[('She ate the Apple.', 65),
 ('The dog at the cat.', 42),
 ('She loves that cat', 30)]
```

