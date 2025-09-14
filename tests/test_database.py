"""Tests for the Data Access Layer (DAL) that interact with a real database."""

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from py_name_entity_normalization.config import Settings
from py_name_entity_normalization.database import dal
from py_name_entity_normalization.database.models import OMOPIndex


@pytest.fixture
def sample_metadata() -> Dict[str, Dict[str, Any]]:
    """Sample metadata for testing."""
    return {"embedding_model_name": {"name": "test-model", "dimension": 4}}


@pytest.fixture
def sample_concepts_df(test_settings: Settings) -> pd.DataFrame:
    """Sample concepts DataFrame for testing."""
    return pd.DataFrame(
        {
            "concept_id": [1, 2, 3],
            "concept_name": ["Aspirin", "Ibuprofen", "Acetaminophen"],
            "domain_id": ["Drug", "Drug", "Condition"],
            "vocabulary_id": ["RxNorm", "RxNorm", "SNOMED"],
            "concept_class_id": ["Ingredient", "Ingredient", "Clinical Finding"],
            "embedding": [
                np.array([0.1, 0.2, 0.3, 0.4]),  # Aspirin
                np.array([0.5, 0.6, 0.7, 0.8]),  # Ibuprofen
                np.array([0.1, 0.2, 0.9, 1.0]),  # Acetaminophen
            ],
        }
    )


def test_upsert_and_get_index_metadata(
    db_session: Session, sample_metadata: Dict[str, Dict[str, Any]]
) -> None:
    """Tests that metadata can be inserted/updated and then retrieved."""
    # Act: Upsert the metadata
    dal.upsert_index_metadata(
        db_session,
        key="embedding_model_name",
        value=sample_metadata["embedding_model_name"],
    )

    # Retrieve and assert
    metadata = dal.get_index_metadata(db_session)
    assert metadata == sample_metadata

    # Update the value and upsert again
    updated_value = {"name": "new-model", "dimension": 5}
    dal.upsert_index_metadata(
        db_session, key="embedding_model_name", value=updated_value
    )
    metadata = dal.get_index_metadata(db_session)
    assert metadata["embedding_model_name"] == updated_value


def test_bulk_insert_and_find_nearest_neighbors(
    db_session: Session, sample_concepts_df: pd.DataFrame
) -> None:
    """Tests bulk inserting concepts and finding their nearest neighbors."""
    # Act: Insert the data
    dal.bulk_insert_omop_concepts(db_session, sample_concepts_df)

    # Check that the data was inserted
    count = db_session.query(OMOPIndex).count()
    assert count == 3

    # Act: Find neighbors for a vector close to "Aspirin"
    query_vector = [0.11, 0.22, 0.33, 0.44]
    candidates = dal.find_nearest_neighbors(db_session, query_vector, k=3)

    # Assert: Results should be ordered by distance
    assert len(candidates) == 3
    assert candidates[0].concept_id == 1  # Aspirin is closest
    assert candidates[1].concept_id == 3  # Acetaminophen is next
    assert candidates[2].concept_id == 2  # Ibuprofen is farthest

    # Check distances are reasonable (pgvector is exact for small N)
    assert candidates[0].distance == pytest.approx(0.0, abs=1e-2)
    assert candidates[0].concept_name == "Aspirin"


def test_find_nearest_neighbors_with_domain_filter(
    db_session: Session, sample_concepts_df: pd.DataFrame
) -> None:
    """Tests that the domain filter is correctly applied."""
    # Arrange: Insert data
    dal.bulk_insert_omop_concepts(db_session, sample_concepts_df)

    # Act: Search with a domain filter
    query_vector = [0.1, 0.2, 0.3, 0.4]
    candidates = dal.find_nearest_neighbors(
        db_session, query_vector, k=5, domains=["Condition"]
    )

    # Assert: Only concepts from the 'Condition' domain should be returned
    assert len(candidates) == 1
    assert candidates[0].concept_id == 3
    assert candidates[0].domain_id == "Condition"


def test_models_repr() -> None:
    """Tests the __repr__ methods of the ORM models."""
    from py_name_entity_normalization.database.models import IndexMetadata

    # Test IndexMetadata
    metadata = IndexMetadata(key="test_key", value={"some": "value"})
    assert (
        repr(metadata) == "<IndexMetadata(key='test_key', value='{'some': 'value'}')>"
    )

    # Test OMOPIndex
    omop_concept = OMOPIndex(concept_id=123, concept_name="Test Concept")
    assert repr(omop_concept) == "<OMOPIndex(concept_id=123, name='Test Concept')>"
