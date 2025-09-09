"""
Tests for the Data Access Layer (DAL).
"""
from unittest.mock import MagicMock, patch

import numpy as np
from py_name_entity_normalization.database import dal
from py_name_entity_normalization.database.models import IndexMetadata, OMOPIndex


def test_find_nearest_neighbors(mock_db_session, test_settings):
    """
    Tests the find_nearest_neighbors function.
    """
    # Arrange: Set up the mock session to return dummy data
    dim = test_settings.EMBEDDING_MODEL_DIMENSION
    mock_result_row = MagicMock()
    mock_result_row.OMOPIndex = OMOPIndex(
        concept_id=1,
        concept_name="Aspirin",
        domain_id="Drug",
        vocabulary_id="RxNorm",
        concept_class_id="Ingredient",
        embedding=np.random.rand(dim).tolist(),
    )
    mock_result_row.distance = 0.1
    mock_db_session.execute.return_value.all.return_value = [mock_result_row]

    # Act: Call the function
    query_vector = np.random.rand(dim).tolist()
    candidates = dal.find_nearest_neighbors(mock_db_session, query_vector, k=5)

    # Assert: Check that the query was constructed correctly and results are parsed
    assert mock_db_session.execute.call_count == 1
    stmt = mock_db_session.execute.call_args[0][0]
    compiled = stmt.compile()
    compiled_stmt_str = str(compiled)
    assert "omop_concept_index" in compiled_stmt_str
    # Check that the order_by clause uses the cosine distance operator
    assert "<=>" in compiled_stmt_str
    assert "LIMIT" in compiled_stmt_str
    # Check that the limit parameter is correct
    assert 5 in compiled.params.values()

    assert len(candidates) == 1
    assert candidates[0].concept_id == 1
    assert candidates[0].concept_name == "Aspirin"
    assert candidates[0].distance == 0.1


def test_find_nearest_neighbors_with_domain_filter(mock_db_session, test_settings):
    """
    Tests that the domain filter is correctly applied in find_nearest_neighbors.
    """
    # Arrange
    mock_db_session.execute.return_value.all.return_value = []

    # Act
    query_vector = np.random.rand(test_settings.EMBEDDING_MODEL_DIMENSION).tolist()
    dal.find_nearest_neighbors(
        mock_db_session, query_vector, k=10, domains=["Drug", "Condition"]
    )

    # Assert
    assert mock_db_session.execute.call_count == 1
    stmt = mock_db_session.execute.call_args[0][0]
    assert "omop_concept_index.domain_id" in str(stmt.whereclause)
    # Check that the parameters are correct without relying on literal binds
    params = stmt.compile().params
    assert "domain_id_1" in params
    assert params["domain_id_1"] == ["Drug", "Condition"]


def test_get_index_metadata(mock_db_session):
    """
    Tests the get_index_metadata function.
    """
    # Arrange
    mock_meta_item = MagicMock()
    mock_meta_item.key = "model_name"
    mock_meta_item.value = {"name": "test-model"}
    mock_db_session.execute.return_value.scalars.return_value.all.return_value = [
        mock_meta_item
    ]

    # Act
    metadata = dal.get_index_metadata(mock_db_session)

    # Assert
    assert metadata == {"model_name": {"name": "test-model"}}
    assert mock_db_session.execute.call_count == 1


def test_upsert_index_metadata(mock_db_session):
    """
    Tests the upsert_index_metadata function.
    """
    # Act
    dal.upsert_index_metadata(
        mock_db_session, key="model", value={"name": "test-model"}
    )

    # Assert
    assert mock_db_session.execute.call_count == 1
    assert mock_db_session.commit.call_count == 1
    stmt = mock_db_session.execute.call_args[0][0]
    assert "index_metadata" in str(stmt)
    # Check that the statement is an insert with an on_conflict_do_update clause
    assert stmt.is_insert
    assert stmt.on_conflict_do_update is not None


def test_bulk_insert_omop_concepts(mock_db_session, mock_pandas_read_csv):
    """
    Tests the bulk_insert_omop_concepts function.
    """
    # Arrange
    df = mock_pandas_read_csv.return_value[0]

    # Act
    dal.bulk_insert_omop_concepts(mock_db_session, df)

    # Assert
    assert mock_db_session.bulk_insert_mappings.call_count == 1
    assert mock_db_session.commit.call_count == 1
    # Check that the mapping and data are correct
    args, kwargs = mock_db_session.bulk_insert_mappings.call_args
    assert args[0] == OMOPIndex
    assert len(args[1]) == len(df)
    assert args[1][0]["concept_id"] == 1
    assert args[1][0]["concept_name"] == "Aspirin"


def test_models_repr():
    """
    Tests the __repr__ methods of the ORM models.
    """
    # Test IndexMetadata
    metadata = IndexMetadata(key="test_key", value={"some": "value"})
    assert (
        repr(metadata) == "<IndexMetadata(key='test_key', value='{'some': 'value'}')>"
    )

    # Test OMOPIndex
    omop_concept = OMOPIndex(concept_id=123, concept_name="Test Concept")
    assert repr(omop_concept) == "<OMOPIndex(concept_id=123, name='Test Concept')>"


@patch("py_name_entity_normalization.database.dal.Base.metadata")
def test_create_database_schema(mock_metadata):
    """
    Tests that create_database_schema calls the underlying SQLAlchemy method.
    """
    # Arrange
    mock_engine = MagicMock()

    # Act
    dal.create_database_schema(mock_engine)

    # Assert
    mock_metadata.create_all.assert_called_once_with(mock_engine)
