Persistence
================

You might want to persist processes to disk and reuses them in production pipelines.
Neofuzz can serialize indexed Process objects for you to a given directory.

You can save indexed processes like so:

.. code-block:: python

   from neofuzz import char_ngram_process
   from neofuzz.tokenization import SubWordVectorizer
 
   process = char_ngram_process()
   process.index(corpus)
 
   process.to_disk("process/")


And then load them in a production environment:

.. code-block:: python

   from neofuzz import Process
 
   process = Process.from_disk("process/")
