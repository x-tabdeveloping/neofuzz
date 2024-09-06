import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
import pynndescent
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import pairwise_distances
from thefuzz import process as thefuzz_process


class Process:
    """TheFuzz-compatible process class for quick searching options.
    Beyond the vectorizer all parameters refer to the approximate nearest
    neighbour search.

    Parameters
    ----------
    vectorizer: sklearn vectorizer
        Some kind of vectorizer model that can vectorize strings.
        You could use tf-idf, bow or even a Pipeline that
        has multiple steps.
    refine_levenshtein: bool, default False
        Indicates whether final results should be refined with the Levenshtein algorithm
    metric: string or callable, default 'cosine'
        The metric to use for computing nearest neighbors. If a callable is
        used it must be a numba njit compiled function. Supported metrics
        include:

        * euclidean
        * manhattan
        * chebyshev
        * minkowski
        * canberra
        * braycurtis
        * mahalanobis
        * wminkowski
        * seuclidean
        * cosine
        * correlation
        * haversine
        * hamming
        * jaccard
        * dice
        * russelrao
        * kulsinski
        * rogerstanimoto
        * sokalmichener
        * sokalsneath
        * yule
        * hellinger
        * wasserstein-1d

        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.
    metric_kwds: dict, default {}
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.
    n_neighbors: int, default 30
        The number of neighbors to use in k-neighbor graph graph_data structure
        used for fast approximate nearest neighbor search. Larger values
        will result in more accurate search results at the cost of
        computation time.
    n_trees: int, default None
        This implementation uses random projection forests for initializing the index
        build process.
        This parameter controls the number of trees in that forest.
        A larger number will result in more accurate neighbor computation
        at the cost of performance.
        The default of None means a value will be chosen based on the
        size of the graph_data.
    leaf_size: int, default None
        The maximum number of points in a leaf for the random projection trees.
        The default of None means a value will be chosen based on n_neighbors.
    pruning_degree_multiplier: float, default 1.5
        How aggressively to prune the graph.
        Since the search graph is undirected
        (and thus includes nearest neighbors and reverse nearest neighbors)
        vertices can have very high degree
        -- the graph will be pruned such that no
        vertex has degree greater than
        ``pruning_degree_multiplier * n_neighbors``.
    diversify_prob: float, default 1.0
        The search graph get "diversified" by removing potentially unnecessary
        edges. This controls the volume of edges removed.
        A value of 0.0 ensures that no edges get removed,
        and larger values result in significantly more
        aggressive edge removal.
        A value of 1.0 will prune all edges that it can.
    tree_init: bool, default True
        Whether to use random projection trees for initialization.
    init_graph: np.ndarray, default None
        2D array of indices of candidate neighbours of the shape
        (data.shape[0], n_neighbours). If the j-th neighbour of the i-th
        instances is unknown, use init_graph[i, j] = -1
    init_dist: np.ndarray, default None
        2D array with the same shape as init_graph,
        such that metric(data[i], data[init_graph[i, j]]) equals
        init_dist[i, j]
    random_state: int, RandomState instance or None, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    algorithm: str, default 'standard'
        This implementation provides an alternative algorithm for
        construction of the k-neighbors graph used as a search index. The
        alternative algorithm can be fast for large ``n_neighbors`` values.
        The``'alternative'`` algorithm has been deprecated and is no longer
        available.
    low_memory: boolean, default True
        Whether to use a lower memory, but more computationally expensive
        approach to index construction.
    max_candidates: int, default None
        Internally each "self-join" keeps a maximum number of candidates (
        nearest neighbors and reverse nearest neighbors) to be considered.
        This value controls this aspect of the algorithm. Larger values will
        provide more accurate search results later, but potentially at
        non-negligible computation cost in building the index. Don't tweak
        this value unless you know what you're doing.
    n_iters: int, default None
        The maximum number of NN-descent iterations to perform. The
        NN-descent algorithm can abort early if limited progress is being
        made, so this only controls the worst case. Don't tweak
        this value unless you know what you're doing. The default of None means
        a value will be chosen based on the size of the graph_data.
    delta: float, default 0.001
        Controls the early abort due to limited progress. Larger values
        will result in earlier aborts, providing less accurate indexes,
        and less accurate searching. Don't tweak this value unless you know
        what you're doing.
    n_jobs: int or None, default None
        The number of parallel jobs to run for neighbors index construction.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    compressed: bool, default False
        Whether to prune out data not needed for searching the index. This will
        result in a significantly smaller index, particularly useful
        for saving,
        but will remove information that might otherwise be useful.
    """

    def __init__(
        self,
        vectorizer,
        refine_levenshtein=False,
        metric="cosine",
        metric_kwds=None,
        n_neighbors=30,
        n_trees=None,
        leaf_size=None,
        pruning_degree_multiplier=1.5,
        diversify_prob=1.0,
        n_search_trees=1,
        tree_init=True,
        init_graph=None,
        init_dist=None,
        random_state=None,
        low_memory=True,
        max_candidates=None,
        n_iters=None,
        delta=0.001,
        n_jobs=None,
        compressed=False,
        parallel_batch_queries=False,
        verbose=False,
    ):
        self.vectorizer = vectorizer
        self.refine_levenshtein = refine_levenshtein
        self.nearest_neighbours_kwargs = {
            "metric": metric,
            "metric_kwds": metric_kwds,
            "n_neighbors": n_neighbors,
            "n_trees": n_trees,
            "leaf_size": leaf_size,
            "pruning_degree_multiplier": pruning_degree_multiplier,
            "diversify_prob": diversify_prob,
            "n_search_trees": n_search_trees,
            "tree_init": tree_init,
            "init_graph": init_graph,
            "init_dist": init_dist,
            "random_state": random_state,
            "low_memory": low_memory,
            "max_candidates": max_candidates,
            "n_iters": n_iters,
            "delta": delta,
            "n_jobs": n_jobs,
            "compressed": compressed,
            "parallel_batch_queries": parallel_batch_queries,
            "verbose": verbose,
        }
        self.nearest_neighbours = None
        self.options = None
        self.metric = metric

    def index(self, options: Iterable[str]):
        """Indexes all options for fast querying.

        Parameters
        ----------
        options: iterable of str
            All options in which we want search.
        """
        self.options = np.array(options)
        try:
            self.vectorizer.fit(self.options)
        except AttributeError:
            print(
                "Vectorizer could not be fitted, we assume it was pretrained."
            )
        dtm = self.vectorizer.transform(self.options)
        self.nearest_neighbours = pynndescent.NNDescent(
            dtm, **self.nearest_neighbours_kwargs
        )
        self.nearest_neighbours.prepare()

    def query(
        self,
        search_terms: Iterable[str],
        limit: int = 10,
        refine_levenshtein: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Searches for the given terms in the options.

        Parameters
        ----------
        search_terms: iterable of str
            Terms to search for.
        limit: int, default 10
            Amount of closest matches to return.
        refine_levenshtein: bool, default None
            Indicates whether results should be refined with Levenshtein distance
            using TheFuzz.
            This can increase the accuracy of your results.
            If not specified, the process's attribute is used.

        Parameters
        ----------
        indices: array of shape (n_search_terms, limit)
            Indices of the closest options to each search term.
        distances: array of shape (n_search_terms, limit)
            Distances from the closest options to each search term.
        """
        if self.options is None or self.nearest_neighbours is None:
            raise ValueError(
                "Options have not been indexed for the process,"
                " please index before querying."
            )
        search_matrix = self.vectorizer.transform(search_terms)
        indices, distances = self.nearest_neighbours.query(
            search_matrix, k=limit
        )
        if refine_levenshtein is None:
            refine_levenshtein = self.refine_levenshtein
        if refine_levenshtein:
            refined_indices = []
            refined_distances = []
            for term, idx in zip(search_terms, indices):
                options = list(self.options[idx])
                res = thefuzz_process.extract(
                    term, options, limit=len(options)
                )
                res_indices = []
                res_dist = []
                for result_term, result_sim in res:
                    res_indices.append(idx[options.index(result_term)])
                    res_dist.append(1 - (result_sim / 100))
                refined_indices.append(res_indices)
                refined_distances.append(res_dist)
            indices = np.stack(refined_indices)
            distances = np.stack(refined_distances)
        return indices, distances

    def extract(
        self,
        query: str,
        choices: Optional[Iterable[str]] = None,
        limit: int = 10,
        refine_levenshtein: Optional[bool] = None,
    ) -> List[Tuple[str, int]]:
        """TheFuzz compatible querying.

        Parameters
        ----------
        query: str
            Query string to search for.
        choices: iterable of str, default None
            Choices to iterate through. If the options are
            already indexed, this parameter is ignored, otherwise
            it will be used for indexing.
        limit: int, default 10
            Number of results to return
        refine_levenshtein: bool, default None
            Indicates whether results should be refined with Levenshtein distance
            using TheFuzz.
            This can increase the accuracy of your results.
            If not specified, the process's attribute is used.

        Returns
        -------
        list of (str, int)
            List of closest terms and their similarity
            to the query term.
        """
        if self.options is None:
            if choices is None:
                raise ValueError(
                    "No options have been indexed,"
                    "and no choices were provided."
                )
            self.index(options=choices)
        indices, distances = self.query(
            [query], limit=limit, refine_levenshtein=refine_levenshtein
        )
        indices = np.ravel(indices)
        distances = np.ravel(distances)
        scores = (1 - distances) * 100
        scores = scores.astype(int)
        result_terms = self.options[indices]  # type: ignore
        return list(zip(result_terms, scores))

    def extractOne(
        self, query: str, choices: Optional[Iterable[str]] = None
    ) -> Tuple[str, int]:
        """TheFuzz compatible extraction of one item.

        Parameters
        ----------
        query: str
            Query string to search for.
        choices: iterable of str, default None
            Choices to iterate through. If the options are
            already indexed, this parameter is ignored, otherwise
            it will be used for indexing.

        Returns
        -------
        result: str
            Closest term to given search term.
        score: int
            Similarity score.
        """
        res = self.extract(query=query, choices=choices, limit=1)[0]
        return res

    def ratio(self, s1: str, s2: str) -> int:
        """Calculates similarity of two strings.

        Parameters
        ----------
        s1: str
            First string.
        s2: str
            Second string.

        Returns
        -------
        int
            Similarity of the two strings (1-100).
        """
        if self.options is None or self.nearest_neighbours is None:
            raise ValueError(
                "Options have not been indexed for the process,"
                " please index before getting ratios."
            )
        v1, v2 = self.vectorizer.transform([s1, s2])
        distance = pairwise_distances(v1, v2, metric=self.metric)
        distance = np.ravel(distance)[0]
        score = (1 - distance) * 100
        return int(score)

    def to_disk(self, filename: Union[str, Path]):
        """Persists indexed process to disk.

        Parameters
        ----------
        filename: str or Path
            File path to save the process to.
            e.g. `process.joblib`
        """
        if self.options is None or self.nearest_neighbours is None:
            warnings.warn(
                "No options were provided and the process is not indexed. Are you sure you want to persist yet?"
            )
        joblib.dump(self, filename)

    @staticmethod
    def from_disk(filename: Union[str, Path]):
        """Loads indexed process from disk.

        Parameters
        ----------
        filename: str or Path
            File path to save the process to.
            e.g. `process.joblib`
        """
        return joblib.load(filename)


def char_ngram_process(
    ngram_range: Tuple[int, int] = (1, 5),
    tf_idf: bool = True,
    metric: str = "cosine",
    refine_levenshtein: bool = False,
) -> Process:
    """Basic character n-gram based fuzzy search process.

    Parameters
    ----------
    ngram_range: tuple of (int, int), default (1,1)
        Lower and upper boundary of n-values for the character
        n-grams.
    tf_idf: bool, default True
        Flag signifying whether the features should be tf-idf weighted.
    metric: str, default 'cosine'
        Distance metric to use for fuzzy search.
    refine_levenshtein: bool, default None
        Indicates whether results should be refined with Levenshtein distance
        using TheFuzz.
        This can increase the accuracy of your results.
        If not specified, the process's attribute is used.

    Returns
    -------
    Process
        Fuzzy search process.
    """
    if tf_idf:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, analyzer="char")
    else:
        vectorizer = CountVectorizer(ngram_range=ngram_range, analyzer="char")
    return Process(
        vectorizer, metric=metric, refine_levenshtein=refine_levenshtein
    )
