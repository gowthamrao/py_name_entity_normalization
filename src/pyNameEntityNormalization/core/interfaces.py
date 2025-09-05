"""
Defines the core interfaces (Abstract Base Classes) for the normalization system.

These interfaces ensure that different components (like embedders and rankers)
can be swapped out with any implementation that adheres to the specified contract.
This is crucial for modularity and extensibility.
"""
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .schemas import Candidate, RankedCandidate


class IEmbedder(ABC):
    """
    Abstract interface for an embedding model.

    This class defines the contract for any model that can generate
    vector embeddings from text.
    """

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """
        Encodes a single string of text into an embedding vector.

        Args:
            text: The input text.

        Returns:
            A NumPy array representing the embedding.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a batch of texts into embedding vectors.

        Args:
            texts: A list of input texts.

        Returns:
            A NumPy array of shape (n_texts, embedding_dimension).
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Returns the name of the underlying embedding model.

        This is crucial for the model consistency check.

        Returns:
            The model name string.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Returns the dimension of the embeddings produced by the model.

        Returns:
            The embedding dimension as an integer.
        """
        raise NotImplementedError


class IRanker(ABC):
    """
    Abstract interface for a re-ranking strategy.

    This class defines the contract for any component that re-scores
    a list of candidates against a query.
    """

    @abstractmethod
    def rank(
        self, query: str, candidates: List[Candidate]
    ) -> List[RankedCandidate]:
        """
        Re-ranks a list of candidates based on a query.

        Args:
            query: The original query text.
            candidates: A list of Candidate objects retrieved from the
                        ANN search stage.

        Returns:
            A list of RankedCandidate objects, sorted by their new
            re-ranking score in descending order.
        """
        raise NotImplementedError
