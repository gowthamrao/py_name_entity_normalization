"""Unit tests for the Data Access Layer (DAL).

These tests mock the database session to test the DAL functions in isolation,
ensuring that the SQLAlchemy query construction is correct without requiring a
live database connection.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from py_name_entity_normalization.database import dal
from py_name_entity_normalization.database.models import OMOPIndex


def test_get_index_metadata() -> None:
    """Test that get_index_metadata correctly constructs a dictionary."""
    # Arrange
    mock_session = MagicMock()
    mock_item_1 = MagicMock()
    mock_item_1.key = "model_name"
    mock_item_1.value = {"name": "bert"}
    mock_item_2 = MagicMock()
    mock_item_2.key = "index_date"
    mock_item_2.value = {"date": "2023-01-01"}

    mock_session.execute.return_value.scalars.return_value.all.return_value = [
        mock_item_1,
        mock_item_2,
    ]

    # Act
    metadata = dal.get_index_metadata(mock_session)

    # Assert
    assert metadata == {
        "model_name": {"name": "bert"},
        "index_date": {"date": "2023-01-01"},
    }
    mock_session.execute.assert_called_once()


def test_upsert_index_metadata() -> None:
    """Test that upsert_index_metadata constructs a correct insert statement."""
    # Arrange
    mock_session = MagicMock()
    key = "test_key"
    value = {"data": "test_value"}

    # Act
    dal.upsert_index_metadata(mock_session, key, value)

    # Assert
    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


def test_bulk_insert_omop_concepts() -> None:
    """Test that bulk_insert_omop_concepts calls bulk_insert_mappings."""
    # Arrange
    mock_session = MagicMock()
    data = {
        "concept_id": [1, 2],
        "concept_name": ["A", "B"],
        "domain_id": ["Drug", "Drug"],
        "vocabulary_id": ["RxNorm", "RxNorm"],
        "concept_class_id": ["Ingredient", "Ingredient"],
        "embedding": [[0.1], [0.2]],
    }
    df = pd.DataFrame(data)

    # Act
    dal.bulk_insert_omop_concepts(mock_session, df)

    # Assert
    mock_session.bulk_insert_mappings.assert_called_once_with(
        OMOPIndex, df.to_dict(orient="records")
    )
    mock_session.commit.assert_called_once()


@patch("py_name_entity_normalization.database.dal.select")
def test_find_nearest_neighbors_no_domain(mock_select: MagicMock) -> None:
    """Test find_nearest_neighbors without a domain filter."""
    # Arrange
    mock_session = MagicMock()
    mock_stmt = MagicMock()
    mock_select.return_value.order_by.return_value.limit.return_value = mock_stmt

    # Mock the row object that SQLAlchemy returns
    mock_row = MagicMock()
    mock_row.OMOPIndex = OMOPIndex(
        concept_id=1,
        concept_name="Test",
        vocabulary_id="Test",
        concept_class_id="Test",
        domain_id="Test",
    )
    mock_row.distance = 0.5
    mock_session.execute.return_value.all.return_value = [mock_row]

    # Act
    candidates = dal.find_nearest_neighbors(mock_session, [0.1], 10)

    # Assert
    assert len(candidates) == 1
    assert candidates[0].concept_id == 1
    assert candidates[0].distance == 0.5
    mock_stmt.where.assert_not_called()


@patch("py_name_entity_normalization.database.dal.select")
def test_find_nearest_neighbors_with_domain(mock_select: MagicMock) -> None:
    """Test that find_nearest_neighbors correctly applies the domain filter."""
    # Arrange
    mock_session = MagicMock()
    mock_stmt_limit = MagicMock()
    mock_stmt_where = MagicMock()
    mock_select.return_value.order_by.return_value.limit.return_value = mock_stmt_limit
    mock_stmt_limit.where.return_value = mock_stmt_where
    mock_session.execute.return_value.all.return_value = []

    # Act
    dal.find_nearest_neighbors(mock_session, [0.1], 10, domains=["Drug"])

    # Assert
    mock_stmt_limit.where.assert_called_once()
