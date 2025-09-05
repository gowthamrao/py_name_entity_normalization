"""
Tests for the embedder module, including the concrete implementation and factory.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyNameEntityNormalization.embedders.factory import get_embedder
from pyNameEntityNormalization.embedders.sentence_transformer import (
    SentenceTransformerEmbedder,
)


@pytest.fixture
def mock_sentence_transformer(mocker, test_settings):
    """
    Mocks the SentenceTransformer class where it is used.
    """
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(
        5, test_settings.EMBEDDING_MODEL_DIMENSION
    )
    mock_model.get_sentence_embedding_dimension.return_value = (
        test_settings.EMBEDDING_MODEL_DIMENSION
    )
    # Patch where the class is imported and used, not its source.
    return mocker.patch(
        "pyNameEntityNormalization.embedders.sentence_transformer.SentenceTransformer",
        return_value=mock_model,
    )


def test_sentence_transformer_embedder_init(
    mock_sentence_transformer, test_settings
):
    """
    Tests the initialization of the SentenceTransformerEmbedder.
    """
    # Act
    embedder = SentenceTransformerEmbedder(model_name=test_settings.EMBEDDING_MODEL_NAME)

    # Assert
    # Check that the underlying model was initialized correctly
    mock_sentence_transformer.assert_called_once_with(
        test_settings.EMBEDDING_MODEL_NAME, device=embedder.device
    )
    assert embedder.model is not None


def test_sentence_transformer_embedder_encode(
    mock_sentence_transformer, test_settings
):
    """
    Tests the encode method.
    """
    # Arrange
    embedder = SentenceTransformerEmbedder(model_name=test_settings.EMBEDDING_MODEL_NAME)
    text = "some text"

    # Act
    embedding = embedder.encode(text)

    # Assert
    embedder.model.encode.assert_called_once_with(
        text, convert_to_numpy=True, device=embedder.device, normalize_embeddings=True
    )
    assert isinstance(embedding, np.ndarray)


def test_sentence_transformer_embedder_encode_batch(
    mock_sentence_transformer, test_settings
):
    """
    Tests the encode_batch method.
    """
    # Arrange
    embedder = SentenceTransformerEmbedder(model_name=test_settings.EMBEDDING_MODEL_NAME)
    texts = ["text 1", "text 2"]

    # Act
    embeddings = embedder.encode_batch(texts)

    # Assert
    embedder.model.encode.assert_called_once_with(
        texts,
        convert_to_numpy=True,
        batch_size=32,
        device=embedder.device,
        normalize_embeddings=True,
    )
    assert isinstance(embeddings, np.ndarray)


def test_sentence_transformer_embedder_getters(
    mock_sentence_transformer, test_settings
):
    """
    Tests the get_model_name and get_dimension methods.
    """
    # Arrange
    embedder = SentenceTransformerEmbedder(model_name=test_settings.EMBEDDING_MODEL_NAME)

    # Act & Assert
    assert embedder.get_model_name() == test_settings.EMBEDDING_MODEL_NAME
    assert embedder.get_dimension() == test_settings.EMBEDDING_MODEL_DIMENSION
    embedder.model.get_sentence_embedding_dimension.assert_called_once()


def test_get_embedder_factory(test_settings):
    """
    Tests the get_embedder factory function.
    """
    # Arrange - Patch the embedder class to avoid real model loading
    with patch(
        "pyNameEntityNormalization.embedders.factory.SentenceTransformerEmbedder"
    ) as mock_embedder_class:
        # Act
        embedder = get_embedder(test_settings)

        # Assert
        mock_embedder_class.assert_called_once_with(
            model_name=test_settings.EMBEDDING_MODEL_NAME
        )
        assert embedder is not None
