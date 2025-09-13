"""
Integration tests for the pyNameEntityNormalization package.

These tests use a real database connection and verify the end-to-end
workflow from the NormalizationEngine to the database.
"""
import pandas as pd
import pytest
from py_name_entity_normalization.core.engine import NormalizationEngine
from py_name_entity_normalization.core.schemas import NormalizationInput
from py_name_entity_normalization.indexer.builder import IndexBuilder


@pytest.fixture(scope="module")
def concept_df():
    """
    Creates a small, controlled DataFrame of concepts for testing.

    The data includes a "clustered" group of similar terms for 'aspirin'
    and another distinct group for 'acetaminophen' to test search precision.
    """
    data = {
        "concept_id": [101, 102, 103, 104, 201, 202, 301],
        "concept_name": [
            "aspirin 81 mg oral tablet",  # Exact match for a query
            "aspirin 325 mg oral tablet",
            "aspirin 81mg tab",  # Close variation
            "Aspirin",
            "acetaminophen 500 mg oral tablet",
            "tylenol extra strength",
            "lisinopril 10 mg tablet",  # A different drug
        ],
        "domain_id": ["Drug", "Drug", "Drug", "Drug", "Drug", "Drug", "Drug"],
        "vocabulary_id": [
            "RxNorm",
            "RxNorm",
            "RxNorm",
            "RxNorm",
            "RxNorm",
            "RxNorm",
            "RxNorm",
        ],
        "concept_class_id": [
            "Clinical Drug",
            "Clinical Drug",
            "Clinical Drug",
            "Ingredient",
            "Clinical Drug",
            "Branded Drug",
            "Clinical Drug",
        ],
        "standard_concept": ["S", "S", "S", "S", "S", "S", "S"],
        "concept_code": ["1", "2", "3", "4", "5", "6", "7"],
        "invalid_reason": [None, None, None, None, None, None, None],
    }
    return pd.DataFrame(data)


def test_normalization_engine_integration(db_session, test_settings, concept_df):
    """
    Tests the full end-to-end normalization pipeline.

    -   Builds an index from a controlled set of concepts.
    -   Initializes the NormalizationEngine.
    -   Performs normalization on a specific term.
    -   Asserts that the top result is the exact match and that other
        semantically similar concepts are returned in the correct order.
    """
    # 1. Build the index with the test data
    # Use a real embedder as specified in the test settings
    builder = IndexBuilder(settings=test_settings, db_session=db_session)
    builder.build_index(concept_df, force=True)

    # 2. Initialize the engine
    # This will perform the model consistency check against the new index
    engine = NormalizationEngine(settings=test_settings)

    # 3. Normalize a specific term that has a close cluster
    norm_input = NormalizationInput(text="aspirin 81 mg oral tablet")
    result = engine.normalize(norm_input)

    # 4. Assert the results
    assert result is not None
    assert len(result.candidates) > 0

    # The top candidate should be the exact match
    top_candidate = result.candidates[0]
    assert top_candidate.concept_id == 101
    assert top_candidate.concept_name == "aspirin 81 mg oral tablet"

    # The rerank score for an exact match should be very high (close to 1.0)
    # This also depends on the ranker, for cosine it's 1.0
    assert top_candidate.rerank_score == pytest.approx(1.0)

    # Verify that other "aspirin" concepts are also in the results
    candidate_ids = {c.concept_id for c in result.candidates}
    assert 102 in candidate_ids
    assert 103 in candidate_ids
    assert 104 in candidate_ids

    # Verify that "acetaminophen" and "lisinopril" are not in the top results
    assert 201 not in candidate_ids
    assert 202 not in candidate_ids
    assert 301 not in candidate_ids

    # Check the order - the next best match should be the close variation
    second_candidate = result.candidates[1]
    assert second_candidate.concept_id == 103  # 'aspirin 81mg tab'
