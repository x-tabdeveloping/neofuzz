What is Neofuzz?
================

Neofuzz is a very fast fuzzy search library, that was developed with TheFuzz (fuzzywuzzy) compatibility in mind.

Why is Neofuzz fast?
^^^^^^^^^^^^^^^^^^^^

Most fuzzy search libraries rely on optimizing the hell out of the same couple of fuzzy search algorithms (Hamming distance, Levenshtein distance).
Sometimes unfortunately due to the complexity of these algorithms, no amount of optimization will get you the speed,
that you want.

Neofuzz makes the realization, that you can't go above a certain speed limit by relying on traditional algorithms, and
uses text vectorization and approximate nearest neighbour search in the vector space to speed up this process.

When it comes to the dilemma of speed versus accuracy, Neofuzz goes full-on speed.

When should I choose Neofuzz?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* You need to do repeated searches in the same corpus.
* Levenshtein and Hamming distance is simply not fast enough.
* You are willing to sacrifice the quality of the results for speed.
* You don't mind that the up-front computation to index a corpus might take time.
* You have very long strings, where other methods would be impractical.
* You want to rely on semantic content.
* You need a drop-in replacement for TheFuzz.

When should I NOT choose Neofuzz?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* The corpus changes all the time, or you only want to do one search in a corpus. (It might still give speed-up in that case though.)
* You value the quality of the results over speed.
* You don't mind slower searches in favor of no indexing.
* You have a small corpus with short strings.
