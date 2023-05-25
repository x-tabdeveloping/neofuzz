Getting Started
==================

Installation
^^^^^^^^^^^^

Neofuzz can be simply installed by installing the PyPI package.

.. code-block::

   pip install neofuzz

Usage
^^^^^^^^^

If you just need a quick and dirty algorithm, that will probably get the job done for you, I recommend
using the Character n-gram process, that comes built in with Neofuzz.

.. tip::
   If you find that this process is good for your use case, just stick with it.

.. code-block:: python

   from neofuzz import char_ngram_process

   # Some corpus of strings you want to search in.
   options: List[str] = []

   # Create a process
   process = char_ngram_process()

   # Index the options, so that searches can be fast
   process.index(options)

   # Then you can use the process the same way as in TheFuzz
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

In case you want to speed things up even more, you need semantic search or better results, you might want
to build a custom Process.
