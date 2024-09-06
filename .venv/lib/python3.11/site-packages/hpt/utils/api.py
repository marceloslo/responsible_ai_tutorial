"""Collection of classes defining the API this package depends on.
"""
from abc import ABC, abstractmethod


class BaseLearner(ABC):
    """Interface for a generic learner (follows sklearn's API)."""

    @abstractmethod
    def predict_proba(self, X):
        """Predict class probabilities for X."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, X, y):
        """Fit learner to the provided features `X` and labels `y`."""
        raise NotImplementedError
