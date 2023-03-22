from typing import Callable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


def get_closest(
    ngram_matrix: np.ndarray, ngram_search_term: np.ndarray, top_n: int
) -> Tuple[np.ndarray, np.ndarray]:
    distances = pairwise_distances(
        ngram_matrix, ngram_search_term, metric="cosine"
    )
    distances = np.ravel(distances)
    closest = np.argpartition(distances, kth=top_n)[:top_n]
    distances = distances[closest]
    sorted = np.argsort(distances)
    return closest[sorted], 1 - distances[sorted]


def create_matcher(
    options: List[str],
    token_level: bool = False,
    ngram_range: Tuple[int, int] = (1, 3),
) -> Callable[[str], List[Tuple[str, float]]]:
    vectorizer = TfidfVectorizer(
        analyzer="word" if token_level else "char", ngram_range=ngram_range
    )
    ngram_matrix = vectorizer.fit_transform(options)

    def match(search_term: str, top_n: int = 10) -> List[Tuple[str, float]]:
        ngram_search_term = vectorizer.transform([search_term])
        closest_indices, similarities = get_closest(
            ngram_matrix, ngram_search_term, top_n=top_n
        )
        result = [
            (options[index], similarity)
            for index, similarity in zip(closest_indices, similarities)
        ]
        return result

    return match
