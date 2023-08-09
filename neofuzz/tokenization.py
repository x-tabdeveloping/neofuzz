import collections
from urllib.request import urlopen

from sklearn.feature_extraction.text import CountVectorizer
from tokenizers import Tokenizer
from tokenizers.models import WordPiece

vocab_urls = {
    "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
    "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/vocab.txt",
    "bert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/vocab.txt",
    "bert-large-cased": "https://huggingface.co/bert-large-cased/resolve/main/vocab.txt",
    "bert-base-multilingual-uncased": (
        "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt"
    ),
    "bert-base-multilingual-cased": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/vocab.txt",
    "bert-base-chinese": "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt",
    "bert-base-german-cased": "https://huggingface.co/bert-base-german-cased/resolve/main/vocab.txt",
    "bert-large-uncased-whole-word-masking": (
        "https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/vocab.txt"
    ),
    "bert-large-cased-whole-word-masking": (
        "https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/vocab.txt"
    ),
    "bert-large-uncased-whole-word-masking-finetuned-squad": (
        "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
    ),
    "bert-large-cased-whole-word-masking-finetuned-squad": (
        "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
    ),
    "bert-base-cased-finetuned-mrpc": (
        "https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/vocab.txt"
    ),
    "bert-base-german-dbmdz-cased": "https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/vocab.txt",
    "bert-base-german-dbmdz-uncased": (
        "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/vocab.txt"
    ),
    "TurkuNLP/bert-base-finnish-cased-v1": (
        "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/vocab.txt"
    ),
    "TurkuNLP/bert-base-finnish-uncased-v1": (
        "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/vocab.txt"
    ),
    "wietsedv/bert-base-dutch-cased": (
        "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/vocab.txt"
    ),
}


def load_vocab(vocab_url: str):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with urlopen(vocab_url) as f:
        tokens = list(f)
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def wordpiece_vectorizer(
    bert_model: str = "bert-base-uncased", **kwargs
) -> CountVectorizer:
    if bert_model not in vocab_urls:
        raise ValueError(f"{bert_model} is not a model we can load.")
    else:
        vocab_url = vocab_urls[bert_model]
    tokenizer = Tokenizer(
        WordPiece(
            vocab=load_vocab(vocab_url),
            unk_token="[UNK]",
            max_input_chars_per_word=None,
        )
    )

    def _tokenize(text: str) -> list[str]:
        output = tokenizer.encode(text)
        return output.tokens

    return CountVectorizer(analyzer=_tokenize, **kwargs)  # type: ignore
