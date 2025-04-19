import json
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
from annoy import AnnoyIndex
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
    metric: str, default 'angular'
        The metric to use for computing nearest neighbors.
        Metric can be "angular", "euclidean", "manhattan", "hamming", or "dot".
    n_jobs: int, default -1
        Number of cores to use for indexing, uses all by default.
    n_trees: int, default 10
        Number of trees to build.
    """

    def __init__(
        self,
        vectorizer,
        refine_levenshtein=False,
        metric="angular",
        n_jobs=-1,
        n_trees=10,
    ):
        self.vectorizer = vectorizer
        self.refine_levenshtein = refine_levenshtein
        self.nearest_neighbours = None
        self.options = None
        self.metric = metric
        self.n_jobs = n_jobs
        self.n_trees = n_trees

    def _get_args(self) -> dict:
        return dict(
            refine_levenshtein=self.refine_levenshtein,
            metric=self.metric,
            n_jobs=self.n_jobs,
            n_trees=self.n_trees,
        )

    def index(self, options: Iterable[str]):
        """Indexes all options for fast querying.

        Parameters
        ----------
        options: iterable of str
            All options in which we want search.
        """
        self.options = np.array(list(options))
        try:
            self.vectorizer.fit(self.options)
        except AttributeError:
            print(
                "Vectorizer could not be fitted, we assume it was pretrained."
            )
        vectors = self.vectorizer.transform(self.options)
        n_dimensions = vectors.shape[1]
        self.nearest_neighbours = AnnoyIndex(n_dimensions, self.metric)
        for i_option, vector in enumerate(vectors):
            self.nearest_neighbours.add_item(i_option, vector)
        self.nearest_neighbours.build(self.n_trees, n_jobs=self.n_jobs)

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
        indices = []
        distances = []
        for query_vector in search_matrix:
            ind, dist = self.nearest_neighbours.get_nns_by_vector(
                query_vector, limit, include_distances=True
            )
            indices.append(ind)
            distances.append(dist)
        indices = np.array(indices)
        distances = np.array(distances)
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

    def to_disk(self, save_dir: Union[str, Path]):
        """Persists indexed process to disk.

        Parameters
        ----------
        save_dir: str or Path
            Directory path to save the process to.
            e.g. `n_gram_process/`
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        np.save(
            save_dir.joinpath("options.npy"),
            self.options,
        )
        joblib.dump(self.vectorizer, save_dir.joinpath("vectorizer.joblib"))
        with save_dir.joinpath("config.json").open("w") as config_file:
            config_file.write(json.dumps(self._get_args()))
        self.nearest_neighbours.save(str(save_dir.joinpath("index.annoy")))

    @classmethod
    def from_disk(cls, save_dir: Union[str, Path]):
        """Loads indexed process from disk.

        Parameters
        ----------
        save_dir: str or Path
            Directory path to load the process from.
            e.g. `process.joblib`
        """
        save_dir = Path(save_dir)
        options = np.load(save_dir.joinpath("options.npy"))
        vectorizer = joblib.load(save_dir.joinpath("vectorizer.joblib"))
        n_dims = vectorizer.transform(["something"]).shape[1]
        with save_dir.joinpath("config.json").open() as config_file:
            config = json.loads(config_file.read())
        index = AnnoyIndex(n_dims, config["metric"])
        index.load(str(save_dir.joinpath("index.annoy")))
        result = cls(vectorizer, **config)
        result.options = options
        result.nearest_neighbours = index
        return result


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
