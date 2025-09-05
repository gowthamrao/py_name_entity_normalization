"""
Tests for the offline indexer module.
"""
from unittest.mock import call, patch

import pytest

from pyNameEntityNormalization.indexer.builder import IndexBuilder


@pytest.fixture
def mock_dal_indexer(mocker):
    """Mocks the DAL functions used by the indexer."""
    # We patch the dal module *within the indexer's namespace*
    return mocker.patch("pyNameEntityNormalization.indexer.builder.dal")


@pytest.fixture
def mock_engine_indexer(mocker):
    """Mocks the SQLAlchemy engine used by the indexer."""
    return mocker.patch("pyNameEntityNormalization.indexer.builder.engine")


def test_index_builder_init(test_settings, mock_embedder, mocker):
    """
    Tests the initialization of IndexBuilder.
    """
    # Mock the factory to return our mock embedder
    mocker.patch(
        "pyNameEntityNormalization.indexer.builder.get_embedder",
        return_value=mock_embedder,
    )
    builder = IndexBuilder(test_settings)
    assert builder.embedder is mock_embedder


def test_index_builder_init_dimension_mismatch(test_settings, mock_embedder, mocker):
    """
    Tests that IndexBuilder raises an error if configured dimension and
    model dimension do not match.
    """
    mock_embedder.get_dimension.return_value = 999  # Different from settings
    mocker.patch(
        "pyNameEntityNormalization.indexer.builder.get_embedder",
        return_value=mock_embedder,
    )

    with pytest.raises(ValueError, match="Configuration error"):
        IndexBuilder(test_settings)


def test_build_index_from_csv(
    test_settings,
    mock_dal_indexer,
    mock_engine_indexer,
    mock_embedder,
    mock_db_session,
    mock_pandas_read_csv,
    mocker,
):
    """
    Tests the full build_index_from_csv workflow.
    """
    # Arrange
    mocker.patch("builtins.open", mocker.mock_open(read_data="data" * 100))
    mocker.patch(
        "pyNameEntityNormalization.indexer.builder.get_embedder",
        return_value=mock_embedder,
    )
    builder = IndexBuilder(test_settings)

    # Act
    builder.build_index_from_csv("dummy_path.csv", session=mock_db_session)

    # Assert
    # 1. Check data processing
    mock_pandas_read_csv.assert_called_once()
    mock_embedder.encode_batch.assert_called_once()

    # 2. Check database writes
    mock_dal_indexer.bulk_insert_omop_concepts.assert_called_once()

    # 3. Check metadata write
    mock_dal_indexer.upsert_index_metadata.assert_called_once_with(
        mock_db_session,
        key="embedding_model_name",
        value={
            "name": "test/dummy-bert",
            "dimension": test_settings.EMBEDDING_MODEL_DIMENSION,
        },
    )

    # 4. Check index creation
    mock_db_session.execute.assert_has_calls([
        call(mocker.ANY), # CREATE EXTENSION
        call(mocker.ANY)  # CREATE INDEX
    ])
    sql_call = str(mock_db_session.execute.call_args_list[1].args[0])
    assert "CREATE INDEX ON omop_concept_index USING hnsw (embedding vector_cosine_ops)" in sql_call
    mock_db_session.commit.assert_called()


def test_build_index_from_csv_with_force(
    test_settings,
    mock_dal_indexer,
    mock_engine_indexer,
    mock_embedder,
    mock_db_session,
    mock_pandas_read_csv,
    mocker,
):
    """
    Tests that the 'force' flag correctly drops and recreates the schema.
    """
    # Arrange
    mocker.patch("builtins.open", mocker.mock_open(read_data="data" * 100))
    mocker.patch(
        "pyNameEntityNormalization.indexer.builder.get_embedder",
        return_value=mock_embedder,
    )
    builder = IndexBuilder(test_settings)

    # Act
    builder.build_index_from_csv("dummy_path.csv", session=mock_db_session, force=True)

    # Assert
    # Check that schema creation/deletion methods were called
    mock_dal_indexer.Base.metadata.drop_all.assert_called_once_with(mock_engine_indexer)
    mock_dal_indexer.create_database_schema.assert_called_once_with(mock_engine_indexer)
