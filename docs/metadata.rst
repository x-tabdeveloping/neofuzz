Handling Metadata
=================

Neofuzz makes it easy to do fuzzy search in text corpora.
Sometimes it is, however beneficial to be able to access metadata about the entries retrieved in fuzzy search.

The most sensible way to handle this is to store your metadata in a table that is in the same order as the corpus.

.. code-block:: python

   import pandas as pd

   corpus: list[str] = [...]
   metadata = pd.DataFrame(...)

   # The tenth element in both corresponds to the same entry
   tenth_text = corpus[9]
   tenth_metadata_entry = metadata.iloc[9]
 
Then you can use the query() method to retrieve indices and distances instead of passages:

.. code-block:: python

   from neofuzz import Process
 
   process = Process(...)
   process.index(corpus)

   # Both results will be arrays shaped (len(search_terms), limit)
   indices, distances = process.query(search_terms=["Search term 1", "Search term 2"], limit=5)

   results_for_term1 = [corpus[idx] for idx in indices[0]]
   metadata_for_term1 = metadata.iloc[indices[0]]

   results_for_term2 = [corpus[idx] for idx in indices[1]]
   metadata_for_term2 = metadata.iloc[indices[1]]
