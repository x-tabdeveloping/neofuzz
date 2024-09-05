<img align="left" width="82" height="82" src="docs/_static/logo.svg">

# Neofuzz

<br>

Blazing fast, lightweight and customizable fuzzy and semantic text search in Python.

## Introduction ([Documentation](https://x-tabdeveloping.github.io/neofuzz/))
Neofuzz is a fuzzy search library based on vectorization and approximate nearest neighbour
search techniques.

### New in version 0.3.0
Now you can reorder your search results using Levenshtein distance!
Sometimes n-gram processes or vectorized processes don't quite order the results correctly.
In these cases you can retrieve a higher number of examples from the indexed corpus, then refine those results with Levenshtein distance.

```python
from neofuzz import char_ngram_process

process = char_ngram_process()
process.index(corpus)

process.extract("your query", limit=30, refine_levenshtein=True)
```

### Why is Neofuzz fast?
Most fuzzy search libraries rely on optimizing the hell out of the same couple of fuzzy search algorithms (Hamming distance, Levenshtein distance). Sometimes unfortunately due to the complexity of these algorithms, no amount of optimization will get you the speed, that you want.

Neofuzz makes the realization, that you can’t go above a certain speed limit by relying on traditional algorithms, and uses text vectorization and approximate nearest neighbour search in the vector space to speed up this process.

When it comes to the dilemma of speed versus accuracy, Neofuzz goes full-on speed.

### When should I choose Neofuzz?
 - You need to do repeated searches in the same corpus.
 - Levenshtein and Hamming distance is simply not fast enough.
 - You are willing to sacrifice the quality of the results for speed.
 - You don’t mind that the up-front computation to index a corpus might take time.
 - You have very long strings, where other methods would be impractical.
 - You want to rely on semantic content.
 - You need a drop-in replacement for TheFuzz.

### When should I NOT choose Neofuzz?
 - The corpus changes all the time, or you only want to do one search in a corpus. (It might still give speed-up in that case though.)
 - You value the quality of the results over speed.
 - You don’t mind slower searches in favor of no indexing.
 - You have a small corpus with short strings.

## [Usage](https://x-tabdeveloping.github.io/neofuzz/getting_started.html)

You can install Neofuzz from PyPI:

```bash
pip install neofuzz
```

If you want a plug-and play experience you can create a generally good quick and dirty
process with the `char_ngram_process()` process.

```python
from neofuzz import char_ngram_process

# We create a process that takes character 1 to 5-grams as features for
# vectorization and uses a tf-idf weighting scheme.
# We will use cosine distance for the nearest neighbour search.
process = char_ngram_process(ngram_range=(1,5), metric="cosine", tf_idf=True)

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

## [Custom Processes](https://x-tabdeveloping.github.io/neofuzz/custom_vectorizer.html)

You can customize Neofuzz’s behaviour by making a custom process.
Under the hood every Neofuzz Process relies on the same two components:

 - A vectorizer, which turns texts into a vectorized form, and can be fully customized.
 - Approximate Nearest Neighbour search, which indexes the vector space and can find neighbours of a given vector very quickly. This component is fixed to be PyNNDescent, but all of its parameters are exposed in the API, so its behaviour can also be altered at will.

### Words as Features

If you’re more interested in the words/semantic content of the text you can also use them as features. This can be very useful especially with longer texts, such as literary works.

```python
from neofuzz import Process
from sklearn.feature_extraction.text import TfidfVectorizer

 # Vectorization with words is the default in sklearn.
 vectorizer = TfidfVectorizer()

 # We use cosine distance because it's waay better for high-dimentional spaces.
 process = Process(vectorizer, metric="cosine")
```

### Dimensionality Reduction

You might find that the speed of your fuzzy search process is not sufficient. In this case it might be desirable to reduce the dimentionality of the produced vectors with some matrix decomposition method or topic model.

Here for example I use NMF (excellent topic model and incredibly fast one too) too speed up my fuzzy search pipeline.

```python
from neofuzz import Process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline

# Vectorization with tokens again
vectorizer = TfidfVectorizer()
# Dimensionality reduction method to 20 dimensions
nmf = NMF(n_components=20)
# Create a pipeline of the two
pipeline = make_pipeline(vectorizer, nmf)

process = Process(pipeline, metric="cosine")
```

### Semantic Search/Large Language Models

With Neofuzz you can easily use semantic embeddings to your advantage, and can use both attention-based language models (Bert), just simple neural word or document embeddings (Word2Vec, Doc2Vec, FastText, etc.) or even OpenAI’s LLMs.

We recommend you try embetter, which has a lot of built-in sklearn compatible vectorizers.
```bash
pip install embetter
```

```python
from embetter.text import SentenceEncoder
from neofuzz import Process

# Here we will use a pretrained Bert sentence encoder as vectorizer
vectorizer = SentenceEncoder("all-distilroberta-v1")
# Then we make a process with the language model
process = Process(vectorizer, metric="cosine")

# Remember that the options STILL have to be indexed even though you have a pretrained vectorizer
process.index(options)
```
