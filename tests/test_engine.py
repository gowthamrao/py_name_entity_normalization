"""
Tests for the NormalizationEngine.
"""
from unittest.mock import MagicMock, patch

import pytest

from pyNameEntityNormalization.core.engine import NormalizationEngine
from pyNameEntityNormalization.core.schemas import NormalizationInput


@pytest.fixture
def mock_dal(mocker):
    """Mocks the data access layer."""
    return mocker.patch("pyNameEntityNormalization.core.engine.dal")


@pytest.fixture
def mock_factories(mocker, mock_embedder, mock_ranker):
    """Mocks the embedder and ranker factories."""
    mocker.patch(
        "pyNameEntityNormalization.core.engine.get_embedder",
        return_value=mock_embedder,
    )
    mocker.patch(
        "pyNameEntityNormalization.core.engine.get_ranker", return_value=mock_ranker
    )


def test_engine_init_success(
    test_settings, mock_dal, mock_factories, mock_db_session
):
    """
    Tests successful initialization of the NormalizationEngine.
    """
    # Arrange: Metadata matches the mock embedder's model name
    mock_dal.get_index_metadata.return_value = {
        "embedding_model_name": {"name": "test/dummy-bert"}
    }

    # Act & Assert: Should initialize without error
    try:
        NormalizationEngine(settings=test_settings)
    except ValueError:
        pytest.fail("NormalizationEngine raised ValueError unexpectedly.")


def test_engine_init_model_mismatch(
    test_settings, mock_dal, mock_factories, mock_db_session
):
    """
    Tests that initialization fails if the model name mismatches.
    """
    # Arrange: Metadata has a different model name
    mock_dal.get_index_metadata.return_value = {
        "embedding_model_name": {"name": "a-different-model"}
    }

    # Act & Assert
    with pytest.raises(ValueError, match="Model mismatch!"):
        NormalizationEngine(settings=test_settings)


def test_engine_init_no_metadata(
    test_settings, mock_dal, mock_factories, mock_db_session
):
    """
    Tests that initialization succeeds if no index metadata is found.
    """
    # Arrange: DAL returns empty metadata
    mock_dal.get_index_metadata.return_value = {}

    # Act & Assert
    try:
        NormalizationEngine(settings=test_settings)
    except ValueError:
        pytest.fail("Engine initialization failed with no metadata.")


def test_normalize_happy_path(
    test_settings, mock_dal, mock_factories, mock_db_session, sample_candidates
):
    """
    Tests a full, successful run of the normalize method.
    """
    # Arrange
    mock_dal.get_index_metadata.return_value = {}  # Skip consistency check
    mock_dal.find_nearest_neighbors.return_value = sample_candidates
    engine = NormalizationEngine(settings=test_settings)

    # Act
    norm_input = NormalizationInput(text="aspirin")
    result = engine.normalize(norm_input)

    # Assert
    # Check that DAL was called correctly
    mock_dal.find_nearest_neighbors.assert_called_once()
    # Check that ranker was called (via the mock_ranker fixture's side_effect)
    assert engine.ranker.rank.call_count == 1
    # Check final output
    assert result.input == norm_input
    # The default threshold of 0.85 filters one candidate (sim=0.2)
    # So only 2 should be left.
    assert len(result.candidates) == 2
    assert result.candidates[0].rerank_score == 0.99


def test_normalize_thresholding(
    test_settings, mock_dal, mock_factories, mock_db_session, sample_candidates
):
    """
    Tests that the confidence threshold is applied correctly.
    """
    # Arrange
    mock_dal.get_index_metadata.return_value = {}
    mock_dal.find_nearest_neighbors.return_value = sample_candidates
    engine = NormalizationEngine(settings=test_settings)

    # One candidate is below the 0.85 threshold (1 - 0.8 = 0.2)
    test_settings.DEFAULT_CONFIDENCE_THRESHOLD = 0.85

    # Act
    result = engine.normalize(NormalizationInput(text="aspirin"))

    # Assert
    # Check that the ranker was called with only the 2 candidates above the threshold
    assert engine.ranker.rank.call_count == 1
    call_args, _ = engine.ranker.rank.call_args
    assert len(call_args[1]) == 2
    assert call_args[1][0].concept_id == 1 # distance 0.1 -> sim 0.9
    assert call_args[1][1].concept_id == 2 # distance 0.05 -> sim 0.95


def test_normalize_no_candidates_found(
    test_settings, mock_dal, mock_factories, mock_db_session
):
    """
    Tests the case where the initial database search returns no candidates.
    """
    # Arrange
    mock_dal.get_index_metadata.return_value = {}
    mock_dal.find_nearest_neighbors.return_value = [] # No candidates
    engine = NormalizationEngine(settings=test_settings)

    # Act
    result = engine.normalize(NormalizationInput(text="something unknown"))

    # Assert
    assert engine.ranker.rank.call_count == 0 # Ranker should not be called
    assert len(result.candidates) == 0
