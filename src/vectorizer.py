import numpy as np
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.base import BaseEstimator, TransformerMixin


class Vectorizer(ABC):
    """Abstract base class for all vectorizers."""

    @abstractmethod
    def fit(self, dataframe):
        pass

    @abstractmethod
    def transform(self, dataframe):
        pass


class BoWVectorizer(Vectorizer):
    def __init__(self):
        self.vocabulary = set()

    def fit(self, dataframe):
        """Fits the BoW models to a DataFrame. Each row should represent a document and have a 'text' column that is a
        list of words."""
        for document in dataframe['text']:
            self.vocabulary.update(document)

    def transform(self, dataframe):
        """Transforms a DataFrame of documents into their BoW representation."""
        return dataframe['text'].apply(lambda document: [document.count(word) for word in self.vocabulary])


class TfIdfVectorizer(Vectorizer):
    def dummy_function(self, doc):
        return doc

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=self.dummy_function,
            preprocessor=self.dummy_function,
            token_pattern=None)

    def fit(self, dataframe):
        """Fits the TF-IDF models to a DataFrame of documents. Each row should represent a document and have a 'text'
        column that is a list of words."""
        corpus = list(dataframe['text'])
        self.vectorizer.fit(corpus)

    def transform(self, dataframe):
        """Transforms a DataFrame of documents into their TF-IDF representation."""
        tfidf_matrix = self.vectorizer.transform(dataframe['text'])
        return tfidf_matrix.toarray().tolist()


class Doc2VecVectorizer(Vectorizer, BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=5, window=2, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, X, y=None):
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X)]
        self.model = Doc2Vec(documents,
                             vector_size=self.vector_size,
                             window=self.window,
                             min_count=self.min_count,
                             workers=self.workers)
        return self

    def transform(self, X):
        vectors = [self.model.infer_vector(doc) for doc in X['text']]
        return vectors


class Word2VecVectorizer(Vectorizer):
    def __init__(self, size=100, window=5, min_count=1, workers=4):
        self.model = None
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def fit(self, dataframe):
        """Fits the Word2Vec models to a DataFrame. Each row should represent a document and have a 'text' column that
        is a list of words."""
        sentences = dataframe['text'].tolist()
        self.model = Word2Vec(sentences, vector_size=self.size, window=self.window, min_count=self.min_count,
                              workers=self.workers)

    def transform(self, dataframe):
        """Transforms a DataFrame of documents into their Word2Vec representation."""

        def document_vector(document):
            vector_sum = sum(abs(self.model.wv[word]) for word in document if word in self.model.wv.key_to_index)
            return vector_sum / len(document) if vector_sum is not None else np.zeros(self.size)

        return dataframe['text'].apply(document_vector)
