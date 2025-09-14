"""Tests for the embedder module, including the concrete implementation and factory."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest_mock import MockerFixture

from py_name_entity_normalization.config import Settings
from py_name_entity_normalization.embedders.factory import get_embedder
from py_name_entity_normalization.embedders.sentence_transformer import (
    SentenceTransformerEmbedder,
)


@pytest.fixture
def mock_sentence_transformer(
    mocker: MockerFixture, test_settings: Settings
) -> MagicMock:
    """Mock the SentenceTransformer class where it is used."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(
        5, test_settings.EMBEDDING_MODEL_DIMENSION
    )
    mock_model.get_sentence_embedding_dimension.return_value = (
        test_settings.EMBEDDING_MODEL_DIMENSION
    )
    # Patch where the class is imported and used, not its source.
    return mocker.patch(
        "py_name_entity_normalization.embedders.sentence_transformer.SentenceTransformer",
        return_value=mock_model,
    )


def test_sentence_transformer_embedder_init(
    mock_sentence_transformer: MagicMock, test_settings: Settings
) -> None:
    """Tests the initialization of the SentenceTransformerEmbedder."""
    # Act
    embedder = SentenceTransformerEmbedder(
        model_name=test_settings.EMBEDDING_MODEL_NAME, device="cpu"
    )

    # Assert
    # Check that the underlying model was initialized correctly
    mock_sentence_transformer.assert_called_once_with(
        test_settings.EMBEDDING_MODEL_NAME, device="cpu"
    )
    assert embedder.model is not None


def test_sentence_transformer_embedder_init_device_auto(
    mocker: MockerFixture,
    mock_sentence_transformer: MagicMock,
    test_settings: Settings,
) -> None:
    """Tests that the device is auto-detected correctly if not provided."""
    # Test with CUDA available
    mocker.patch("torch.cuda.is_available", return_value=True)
    embedder = SentenceTransformerEmbedder(
        model_name=test_settings.EMBEDDING_MODEL_NAME
    )
    assert embedder.device == "cuda"
    mock_sentence_transformer.assert_called_with(
        test_settings.EMBEDDING_MODEL_NAME, device="cuda"
    )

    # Test with CUDA not available
    mocker.patch("torch.cuda.is_available", return_value=False)
    embedder = SentenceTransformerEmbedder(
        model_name=test_settings.EMBEDDING_MODEL_NAME
    )
    assert embedder.device == "cpu"
    mock_sentence_transformer.assert_called_with(
        test_settings.EMBEDDING_MODEL_NAME, device="cpu"
    )


def test_sentence_transformer_embedder_encode(
    mock_sentence_transformer: MagicMock, test_settings: Settings
) -> None:
    """Tests the encode method."""
    # Arrange
    embedder = SentenceTransformerEmbedder(
        model_name=test_settings.EMBEDDING_MODEL_NAME
    )
    text = "some text"

    # Act
    embedding = embedder.encode(text)

    # Assert
    embedder.model.encode.assert_called_once_with(
        text, convert_to_numpy=True, device=embedder.device, normalize_embeddings=True
    )
    assert isinstance(embedding, np.ndarray)


def test_sentence_transformer_embedder_encode_batch(
    mock_sentence_transformer: MagicMock, test_settings: Settings
) -> None:
    """Tests the encode_batch method."""
    # Arrange
    embedder = SentenceTransformerEmbedder(
        model_name=test_settings.EMBEDDING_MODEL_NAME
    )
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
    mock_sentence_transformer: MagicMock, test_settings: Settings
) -> None:
    """Tests the get_model_name and get_dimension methods."""
    # Arrange
    embedder = SentenceTransformerEmbedder(
        model_name=test_settings.EMBEDDING_MODEL_NAME
    )

    # Act & Assert
    assert embedder.get_model_name() == test_settings.EMBEDDING_MODEL_NAME
    assert embedder.get_dimension() == test_settings.EMBEDDING_MODEL_DIMENSION
    embedder.model.get_sentence_embedding_dimension.assert_called_once()


def test_get_embedder_factory(test_settings: Settings) -> None:
    """Tests the get_embedder factory function."""
    # Arrange - Patch the embedder class to avoid real model loading
    with patch(
        "py_name_entity_normalization.embedders.factory.SentenceTransformerEmbedder"
    ) as mock_embedder_class:
        # Act
        embedder = get_embedder(test_settings)

        # Assert
        mock_embedder_class.assert_called_once_with(
            model_name=test_settings.EMBEDDING_MODEL_NAME
        )
        assert embedder is not None
