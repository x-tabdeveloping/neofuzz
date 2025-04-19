Custom Processes
================

You can customize Neofuzz's behaviour by making a custom process.

Under the hood every Neofuzz Process relies on the same two components:

* A vectorizer, which turns texts into a vectorized form, and can be fully customized.
* Approximate Nearest Neighbour search, which indexes the vector space and can find neighbours of a given vector very quickly. This component is fixed to be PyNNDescent, but all of its parameters are exposed in the API, so its behaviour can also be altered at will.

The Character N-gram Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default process in Neofuzz is the character n-gram process, and it relies on vectorizing the text in such a manner,
that n-grams become the different features of the text. Plus if you want you can apply a tf-idf weighting scheme, which
makes more specific features (features with more variance) more important, and you can choose a distance measure.

This behaviour is desirable when you have texts that are farily short, don't contain many words, and you
don't want to rely on semantic content.

This piece of code I literally took from the library itself because it's only nine lines.

.. code-block:: python

  from neofuzz import Process
  from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

  def char_ngram_process(
      ngram_range: Tuple[int, int] = (1, 5),
      tf_idf: bool = True,
      metric: str = "angular",
  ) -> Process:
      if tf_idf:
          vectorizer = TfidfVectorizer(ngram_range=ngram_range, analyzer="char")
      else:
          vectorizer = CountVectorizer(ngram_range=ngram_range, analyzer="char")
      return Process(vectorizer, metric=metric)

We use scikit-learn's built-in vectorizer classes, because they already did a great job implementing these.
If you want to know more about what these do, you should check out scikit-learn's docs.

Words as Features
^^^^^^^^^^^^^^^^^

If you're more interested in the words/semantic content of the text you can also use them as features.
This can be very useful especially with longer texts, such as literary works.

.. code-block:: python

  from neofuzz import Process
  from sklearn.feature_extraction.text import TfidfVectorizer

   # Vectorization with words is the default in sklearn.
   vectorizer = TfidfVectorizer()

   # We use cosine distance because it's waay better for high-dimensional spaces.
   process = Process(vectorizer, metric="angular")


Subword Features (New in 0.2.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You might want to utilize subword features in your pipelines, that are a bit more informative than character n-grams.
A good option for this is to use a pretrained tokenizer from a language model!

Here's an example of how to use a Bert-type WordPiece tokenizer for vectorization:

.. code-block:: python

  from neofuzz import Process
  from neofuzz.tokenization import SubWordVectorizer

   # We can use bert's wordpiece tokenizer for feature extraction
   vectorizer = SubWordVectorizer("bert-base-uncased")

   process = Process(vectorizer, metric="angular")


Dimensionality Reduction
^^^^^^^^^^^^^^^^^^^^^^^^

You might find that the speed of your fuzzy search process is not sufficient. In this case it might be desirable to
reduce the dimensionality of the produced vectors with some matrix decomposition method or topic model.

Here for example I use NMF (excellent topic model and incredibly fast one too) too speed up my fuzzy search pipeline.

.. code-block:: python

   from neofuzz import Process
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.decomposition import NMF
   from sklear.pipeline import make_pipeline

   # Vectorization with tokens again
   vectorizer = TfidfVectorizer()
   # Dimensionality reduction method to 20 dimensions
   nmf = NMF(n_components=20)
   # Create a pipeline of the two
   pipeline = make_pipeline(vectorizer, nmf)

   process = Process(pipeline, metric="angular")

Semantic Search
^^^^^^^^^^^^^^^

With Neofuzz you can easily use semantic embeddings to your advantage, and can use both attention-based language models (Bert),
just simple neural word or document embeddings (Word2Vec, Doc2Vec, FastText, etc.) or even OpenAI's LLMs.

We recommend you try embetter, which has a lot of built-in sklearn compatible vectorizers.

.. code-block:: bash

   pip install embetter[text]

.. code-block:: python

   from embetter.text import SentenceEncoder
   from neofuzz import Process

   # Here we will use a pretrained Bert sentence encoder as vectorizer
   vectorizer = SentenceEncoder("all-distilroberta-v1")
   # Then we make a process with the language model
   process = Process(vectorizer, metric="angular")

   # Remember that the options STILL have to be indexed even though you have a pretrained vectorizer
   process.index(options)

