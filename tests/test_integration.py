import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline

from neofuzz import Process
from neofuzz.tokenization import SubWordVectorizer


def test_intergration():
    newsgroups = fetch_20newsgroups(subset="all")
    corpus = newsgroups.data[:2000]
    queries = [
        "This is a query",
        "This is another query",
    ]
    process = Process(make_pipeline(SubWordVectorizer(), NMF(20)))
    process.index(corpus[:-100])
    indices, distances = process.query(queries, limit=10)

    process.to_disk("test_process/")
    process = Process.from_disk("test_process/")

    new_indices, new_distances = process.query(queries)
    assert np.all(
        indices == new_indices
    ), "Indices don't match after persisting model."
    assert np.allclose(
        distances, new_distances
    ), "Distances don't match after persisting model."
