"""
Tests for the ranker modules, including concrete implementations and the factory.
"""
from unittest.mock import MagicMock

import pytest

from pyNameEntityNormalization.rankers.cosine import CosineSimilarityRanker
from pyNameEntityNormalization.rankers.cross_encoder import CrossEncoderRanker
from pyNameEntityNormalization.rankers.factory import get_ranker
from pyNameEntityNormalization.rankers.llm import LLMRanker


def test_cosine_similarity_ranker(sample_candidates):
    """
    Tests the CosineSimilarityRanker.
    """
    # Arrange
    ranker = CosineSimilarityRanker()

    # Act
    ranked = ranker.rank("query", sample_candidates)

    # Assert
    assert len(ranked) == 3
    # Check that scores are 1 - distance
    assert ranked[0].rerank_score == pytest.approx(1.0 - 0.05)  # Highest score
    assert ranked[1].rerank_score == pytest.approx(1.0 - 0.1)
    assert ranked[2].rerank_score == pytest.approx(1.0 - 0.8)  # Lowest score
    # Check that they are sorted by score descending
    assert ranked[0].concept_id == 2
    assert ranked[1].concept_id == 1
    assert ranked[2].concept_id == 3


def test_cross_encoder_ranker(mocker, test_settings, sample_candidates):
    """
    Tests the CrossEncoderRanker, mocking the underlying model.
    """
    # Arrange
    mock_cross_encoder_model = MagicMock()
    # Mock scores, in a different order than the original candidates
    mock_cross_encoder_model.predict.return_value = [0.5, 0.9, 0.2]
    mocker.patch(
        "pyNameEntityNormalization.rankers.cross_encoder.CrossEncoder",
        return_value=mock_cross_encoder_model,
    )

    ranker = CrossEncoderRanker(
        model_name=test_settings.CROSS_ENCODER_MODEL_NAME, device="cpu"
    )

    # Act
    ranked = ranker.rank("Aspirin", sample_candidates)

    # Assert
    # Check that the predict method was called with the correct pairs
    expected_pairs = [
        ("Aspirin", "Aspirin"),
        ("Aspirin", "Aspirin 81mg"),
        ("Aspirin", "Tylenol"),
    ]
    mock_cross_encoder_model.predict.assert_called_once_with(
        expected_pairs, convert_to_numpy=True
    )

    # Check that the results are sorted by the new scores
    assert len(ranked) == 3
    assert ranked[0].concept_id == 2  # score 0.9
    assert ranked[1].concept_id == 1  # score 0.5
    assert ranked[2].concept_id == 3  # score 0.2


def test_cross_encoder_ranker_init_device_auto(mocker, test_settings):
    """
    Tests that the device is auto-detected correctly if not provided.
    """
    mock_cross_encoder_model = MagicMock()
    mock_cross_encoder_constructor = mocker.patch(
        "pyNameEntityNormalization.rankers.cross_encoder.CrossEncoder",
        return_value=mock_cross_encoder_model,
    )

    # Test with CUDA available
    mocker.patch("torch.cuda.is_available", return_value=True)
    ranker = CrossEncoderRanker(model_name=test_settings.CROSS_ENCODER_MODEL_NAME)
    assert ranker.device == "cuda"
    mock_cross_encoder_constructor.assert_called_with(
        test_settings.CROSS_ENCODER_MODEL_NAME, max_length=512, device="cuda"
    )

    # Test with CUDA not available
    mocker.patch("torch.cuda.is_available", return_value=False)
    ranker = CrossEncoderRanker(model_name=test_settings.CROSS_ENCODER_MODEL_NAME)
    assert ranker.device == "cpu"
    mock_cross_encoder_constructor.assert_called_with(
        test_settings.CROSS_ENCODER_MODEL_NAME, max_length=512, device="cpu"
    )


def test_cross_encoder_ranker_empty_candidates(mocker, test_settings):
    """
    Tests that the ranker handles an empty list of candidates gracefully.
    """
    # Arrange
    mock_cross_encoder_model = MagicMock()
    mocker.patch(
        "pyNameEntityNormalization.rankers.cross_encoder.CrossEncoder",
        return_value=mock_cross_encoder_model,
    )
    ranker = CrossEncoderRanker(model_name=test_settings.CROSS_ENCODER_MODEL_NAME)

    # Act
    ranked = ranker.rank("query", [])

    # Assert
    assert ranked == []
    mock_cross_encoder_model.predict.assert_not_called()


def test_llm_ranker():
    """
    Tests that the LLMRanker raises NotImplementedError.
    """
    # Arrange
    ranker = LLMRanker()

    # Act & Assert
    with pytest.raises(NotImplementedError):
        ranker.rank("query", [])


def test_get_ranker_factory(test_settings, mocker):
    """
    Tests the get_ranker factory function for all strategies.
    """
    # Mock the ranker classes to prevent model loading
    mock_cosine = mocker.patch("pyNameEntityNormalization.rankers.factory.CosineSimilarityRanker")
    mock_cross = mocker.patch("pyNameEntityNormalization.rankers.factory.CrossEncoderRanker")
    mock_llm = mocker.patch("pyNameEntityNormalization.rankers.factory.LLMRanker")

    # Test Cosine strategy
    test_settings.RERANKING_STRATEGY = "cosine"
    get_ranker(test_settings)
    mock_cosine.assert_called_once()

    # Test Cross-Encoder strategy
    test_settings.RERANKING_STRATEGY = "cross_encoder"
    get_ranker(test_settings)
    mock_cross.assert_called_once_with(model_name=test_settings.CROSS_ENCODER_MODEL_NAME)

    # Test LLM strategy
    test_settings.RERANKING_STRATEGY = "llm"
    get_ranker(test_settings)
    mock_llm.assert_called_once()


def test_get_ranker_factory_unknown_strategy(test_settings):
    """
    Tests that the factory raises an error for an unknown strategy.
    """
    test_settings.RERANKING_STRATEGY = "unknown_strategy"
    with pytest.raises(ValueError, match="Unknown re-ranking strategy"):
        get_ranker(test_settings)
