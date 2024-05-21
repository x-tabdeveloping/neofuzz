from sklearn.feature_extraction.text import CountVectorizer
from tokenizers import Tokenizer


class SubWordVectorizer(CountVectorizer):
    """Vectorizer that encodes subword features.

    Inhertits from CountVectorizer from sklearn.
    Consult scikit-learn's documentation for more detail on the parameters.

    Parameters
    ----------
    from_model: str, default 'bert-base-uncased'
        Uses tokenizer from the specified language model.
        Should be available on HuggingFace Hub.
    """

    def __init__(
        self,
        from_model: str = "bert-base-uncased",
        input="content",
        encoding="utf-8",
        decode_error="strict",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
    ):
        self._hf_tokenizer = Tokenizer.from_pretrained(from_model)
        self.from_model = from_model

        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            lowercase=False,
            tokenizer=self._tokenize,
        )

    def _tokenize(self, text: str) -> list[str]:
        output = self._hf_tokenizer.encode(text)
        return output.tokens
