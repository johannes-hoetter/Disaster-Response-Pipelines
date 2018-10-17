from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import re

class MessageLengthTransformer(BaseEstimator, TransformerMixin):
    """
    Calculates the length of a single message.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([len(x) for x in X]).reshape(-1, 1)


class SpecialCharacterCounter(BaseEstimator, TransformerMixin):
    """
    Counts any special Character in a message.
    Idea: maybe desperate or urgent messages contain more of those (e.g. exclamation marks etc.)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([len(re.sub('[ a-zA-Z0-9]', '', x)) for x in X]).reshape(-1, 1)