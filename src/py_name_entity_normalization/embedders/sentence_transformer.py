"""Concrete implementation of the IEmbedder interface using sentence-transformers."""

from typing import List, cast

import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from ..core.interfaces import IEmbedder


class SentenceTransformerEmbedder(IEmbedder):
    """An embedder that uses sentence-transformers for embeddings."""

    def __init__(self, model_name: str, device: str | None = None):
        """Initialize the SentenceTransformerEmbedder.

        Args:
        ----
            model_name: The name of the model to load from Hugging Face Hub.
            device: The device to run the model on (e.g., 'cpu', 'cuda').
                    If None, it will auto-detect CUDA availability.

        """
        self._model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading embedding model '{self._model_name}'...")
        print(f"Using device: '{self.device}'")
        self.model = SentenceTransformer(self._model_name, device=self.device)
        print("Model loaded successfully.")

    def encode(self, text: str) -> NDArray[np.float_]:
        """Encode a single string of text into an embedding vector.

        Args:
        ----
            text: The input text.

        Returns:
        -------
            A NumPy array representing the embedding.

        """
        return cast(
            NDArray[np.float_],
            self.model.encode(
                text,
                convert_to_numpy=True,
                device=self.device,
                normalize_embeddings=True,
            ),
        )

    def encode_batch(self, texts: List[str]) -> NDArray[np.float_]:
        """Encode a batch of texts into embedding vectors.

        Args:
        ----
            texts: A list of input texts.

        Returns:
        -------
            A NumPy array of shape (n_texts, embedding_dimension).

        """
        return cast(
            NDArray[np.float_],
            self.model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=32,  # A reasonable default batch size
                device=self.device,
                normalize_embeddings=True,
            ),
        )

    def get_model_name(self) -> str:
        """Return the name of the underlying embedding model.

        Returns
        -------
            The model name string.

        """
        return self._model_name

    def get_dimension(self) -> int:
        """Return the dimension of the embeddings produced by the model.

        Returns
        -------
            The embedding dimension as an integer.

        """
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError("Could not determine embedding dimension from the model.")
        return int(dimension)
