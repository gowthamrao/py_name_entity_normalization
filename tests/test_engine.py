"""Tests for the NormalizationEngine."""

from typing import List, cast
from unittest.mock import MagicMock

import pytest
from py_name_entity_normalization.config import Settings
from py_name_entity_normalization.core.engine import NormalizationEngine
from py_name_entity_normalization.core.schemas import Candidate, NormalizationInput
from py_name_entity_normalization.rankers.cosine import CosineSimilarityRanker
from py_name_entity_normalization.rankers.factory import get_ranker
from py_name_entity_normalization.rankers.llm import LLMRanker
from pytest_mock import MockerFixture


@pytest.fixture
def mock_dal(mocker: MockerFixture) -> MagicMock:
    """Mock the data access layer."""
    return cast(MagicMock, mocker.patch("py_name_entity_normalization.core.engine.dal"))


@pytest.fixture
def mock_factories(
    mocker: MockerFixture, mock_embedder: MagicMock, mock_ranker: MagicMock
) -> None:
    """Mock the embedder and ranker factories."""
    mocker.patch(
        "py_name_entity_normalization.core.engine.get_embedder",
        return_value=mock_embedder,
    )
    mocker.patch(
        "py_name_entity_normalization.core.engine.get_ranker", return_value=mock_ranker
    )


@pytest.fixture
def candidates_with_ambiguous_domain() -> List[Candidate]:
    """Return a list of candidates where 'cold' could be a Drug or a Condition."""
    return [
        Candidate(
            concept_id=100,
            concept_name="Common Cold",
            domain_id="Condition",
            vocabulary_id="SNOMED",
            concept_class_id="Clinical Finding",
            distance=0.1,
        ),
        Candidate(
            concept_id=200,
            concept_name="Cold medicine",
            domain_id="Drug",
            vocabulary_id="RxNorm",
            concept_class_id="Branded Drug",
            distance=0.2,
        ),
    ]


def test_engine_init_success(
    test_settings: Settings,
    mock_dal: MagicMock,
    mock_factories: None,
    mock_db_session: MagicMock,
) -> None:
    """Test successful initialization of the NormalizationEngine."""
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
    test_settings: Settings,
    mock_dal: MagicMock,
    mock_factories: None,
    mock_db_session: MagicMock,
) -> None:
    """Test that initialization fails if the model name mismatches."""
    # Arrange: Metadata has a different model name
    mock_dal.get_index_metadata.return_value = {
        "embedding_model_name": {"name": "a-different-model"}
    }

    # Act & Assert
    with pytest.raises(ValueError, match="Model mismatch!"):
        NormalizationEngine(settings=test_settings)


def test_engine_init_no_metadata(
    test_settings: Settings,
    mock_dal: MagicMock,
    mock_factories: None,
    mock_db_session: MagicMock,
) -> None:
    """Test that initialization succeeds if no index metadata is found."""
    # Arrange: DAL returns empty metadata
    mock_dal.get_index_metadata.return_value = {}

    # Act & Assert
    try:
        NormalizationEngine(settings=test_settings)
    except ValueError:
        pytest.fail("Engine initialization failed with no metadata.")


def test_normalize_happy_path(
    test_settings: Settings,
    mock_dal: MagicMock,
    mock_factories: None,
    mock_db_session: MagicMock,
    comprehensive_candidates: List[Candidate],
) -> None:
    """Test a full, successful run of the normalize method."""
    # Arrange
    mock_dal.get_index_metadata.return_value = {}  # Skip consistency check
    mock_dal.find_nearest_neighbors.return_value = comprehensive_candidates
    engine = NormalizationEngine(settings=test_settings)

    # Act
    norm_input = NormalizationInput(text="aspirin", domains=None)
    result = engine.normalize(norm_input)

    # Assert
    mock_dal.find_nearest_neighbors.assert_called_once()
    engine.ranker.rank.assert_called_once()  # type: ignore
    assert result.input == norm_input
    # Default threshold is 0.85, so 1 - distance must be > 0.85
    # distance must be < 0.15
    # Candidates with distance > 0.15 are filtered out
    # 0.7, 0.2, 0.25, 0.9 are filtered
    assert len(result.candidates) == 6
    assert result.candidates[0].rerank_score == 0.99


def test_normalize_thresholding(
    test_settings: Settings,
    mock_dal: MagicMock,
    mock_factories: None,
    mock_db_session: MagicMock,
    comprehensive_candidates: List[Candidate],
) -> None:
    """Test that the confidence threshold is applied correctly."""
    # Arrange
    mock_dal.get_index_metadata.return_value = {}
    mock_dal.find_nearest_neighbors.return_value = comprehensive_candidates
    engine = NormalizationEngine(settings=test_settings)

    # distances: 0.01, 0.05, 0.1, 0.7, 0.2, 0.25, 0.15, 0.9, 0.12, 0.08
    # similarities: 0.99, 0.95, 0.9, 0.3, 0.8, 0.75, 0.85, 0.1, 0.88, 0.92
    test_settings.DEFAULT_CONFIDENCE_THRESHOLD = 0.91

    # Act
    _ = engine.normalize(NormalizationInput(text="aspirin", domains=None))

    # Assert
    # Ranker is called with candidates with similarity > 0.91
    # Similarities: 0.99, 0.95, 0.92
    engine.ranker.rank.assert_called_once()  # type: ignore
    call_args, _ = engine.ranker.rank.call_args  # type: ignore
    assert len(call_args[1]) == 3
    assert call_args[1][0].concept_id == 1  # sim 0.99
    assert call_args[1][1].concept_id == 2  # sim 0.95
    assert call_args[1][2].concept_id == 10  # sim 0.92


def test_normalize_no_candidates_found(
    test_settings: Settings,
    mock_dal: MagicMock,
    mock_factories: None,
    mock_db_session: MagicMock,
) -> None:
    """Test the case where the initial database search returns no candidates."""
    # Arrange
    mock_dal.get_index_metadata.return_value = {}
    mock_dal.find_nearest_neighbors.return_value = []  # No candidates
    engine = NormalizationEngine(settings=test_settings)

    # Act
    result = engine.normalize(
        NormalizationInput(text="something unknown", domains=None)
    )

    # Assert
    engine.ranker.rank.assert_not_called()  # type: ignore
    assert len(result.candidates) == 0


@pytest.mark.parametrize("text_input", ["", "   ", "!!@#$%"])
def test_normalize_empty_input_text(
    test_settings: Settings,
    mock_dal: MagicMock,
    mock_factories: None,
    mock_db_session: MagicMock,
    text_input: str,
) -> None:
    """Test that empty or cleaned-to-empty input returns an empty list."""
    # Arrange
    mock_dal.get_index_metadata.return_value = {}
    engine = NormalizationEngine(settings=test_settings)

    # Act
    result = engine.normalize(NormalizationInput(text=text_input, domains=None))

    # Assert
    engine.embedder.encode.assert_not_called()  # type: ignore
    mock_dal.find_nearest_neighbors.assert_not_called()
    engine.ranker.rank.assert_not_called()  # type: ignore
    assert len(result.candidates) == 0


def test_engine_with_llm_ranker_fails(
    test_settings: Settings,
    mock_dal: MagicMock,
    mock_embedder: MagicMock,
    mock_db_session: MagicMock,
    comprehensive_candidates: List[Candidate],
    mocker: MockerFixture,
) -> None:
    """Test that the engine correctly uses the LLMRanker and fails as expected."""
    # Arrange
    test_settings.RERANKING_STRATEGY = "llm"
    mock_dal.get_index_metadata.return_value = {}  # Skip consistency check
    mock_dal.find_nearest_neighbors.return_value = comprehensive_candidates

    # We need to patch the factory here to return a real LLMRanker
    mocker.patch(
        "py_name_entity_normalization.core.engine.get_embedder",
        return_value=mock_embedder,
    )
    mocker.patch(
        "py_name_entity_normalization.core.engine.get_ranker",
        side_effect=get_ranker,  # Use the real factory
    )

    engine = NormalizationEngine(settings=test_settings)
    assert isinstance(engine.ranker, LLMRanker)

    # Act & Assert
    with pytest.raises(NotImplementedError, match="LLM-based re-ranking"):
        engine.normalize(NormalizationInput(text="aspirin", domains=None))


def test_normalize_with_domain_filter(
    test_settings: Settings,
    mock_dal: MagicMock,
    mock_factories: None,
    mock_db_session: MagicMock,
    candidates_with_ambiguous_domain: List[Candidate],
) -> None:
    """Test that the domain filter is correctly applied and passed to the DAL."""
    # Arrange
    # Mock DAL to return only the 'Condition' candidate when filtered
    condition_candidate = [candidates_with_ambiguous_domain[0]]
    mock_dal.get_index_metadata.return_value = {}
    mock_dal.find_nearest_neighbors.return_value = condition_candidate
    engine = NormalizationEngine(settings=test_settings)

    # Act
    norm_input = NormalizationInput(text="cold", domains=["Condition"])
    result = engine.normalize(norm_input)

    # Assert
    # Check that find_nearest_neighbors was called with the domain filter
    mock_dal.find_nearest_neighbors.assert_called_once()
    call_args, _ = mock_dal.find_nearest_neighbors.call_args
    assert call_args[3] == ["Condition"]  # domains is the 4th argument

    # Check that the result contains only the condition
    assert len(result.candidates) == 1
    assert result.candidates[0].concept_id == 100
    assert result.candidates[0].domain_id == "Condition"


def test_engine_with_cosine_ranker(
    test_settings: Settings,
    mock_dal: MagicMock,
    mock_embedder: MagicMock,
    mock_db_session: MagicMock,
    comprehensive_candidates: List[Candidate],
    mocker: MockerFixture,
) -> None:
    """Test that the engine works correctly with the CosineSimilarityRanker."""
    # Arrange
    test_settings.RERANKING_STRATEGY = "cosine"
    test_settings.DEFAULT_CONFIDENCE_THRESHOLD = 0.0  # Ensure all candidates are ranked
    mock_dal.get_index_metadata.return_value = {}  # Skip consistency check
    mock_dal.find_nearest_neighbors.return_value = comprehensive_candidates

    mocker.patch(
        "py_name_entity_normalization.core.engine.get_embedder",
        return_value=mock_embedder,
    )
    mocker.patch(
        "py_name_entity_normalization.core.engine.get_ranker",
        side_effect=get_ranker,  # Use the real factory
    )

    engine = NormalizationEngine(settings=test_settings)
    assert isinstance(engine.ranker, CosineSimilarityRanker)

    # Act
    result = engine.normalize(NormalizationInput(text="aspirin", domains=None))

    # Assert
    assert len(result.candidates) == len(comprehensive_candidates)
    # Check that scores are 1 - distance and sorted
    assert result.candidates[0].rerank_score == pytest.approx(1.0 - 0.01)
    assert result.candidates[0].concept_id == 1
    assert result.candidates[-1].rerank_score == pytest.approx(1.0 - 0.9)
    assert result.candidates[-1].concept_id == 8
