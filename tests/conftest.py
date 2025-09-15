"""Pytest fixtures for the entire test suite.

This file contains shared fixtures used across multiple test files, such as
mocked services, database sessions, and configuration objects. This approach
promotes code reuse and makes tests cleaner and easier to maintain.
"""

from typing import Any, Generator, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from py_name_entity_normalization.config import Settings
from py_name_entity_normalization.core.interfaces import IEmbedder, IRanker
from py_name_entity_normalization.core.schemas import Candidate, RankedCandidate
from py_name_entity_normalization.database import dal
from py_name_entity_normalization.database.models import Base
from pytest_mock import MockerFixture
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


@pytest.fixture(scope="function")
def test_settings() -> Settings:
    """Return a Settings object for testing."""
    # Match the default model dimension to avoid pgvector errors during testing
    # when the test settings are used to configure a model but the default settings
    # were used when the ORM model was defined.
    return Settings(  # type: ignore
        DATABASE_URL="postgresql+psycopg://user:password@localhost:5432/nen_db_test",
        EMBEDDING_MODEL_NAME="test/dummy-bert",
        CROSS_ENCODER_MODEL_NAME="test/dummy-cross-encoder",
        EMBEDDING_MODEL_DIMENSION=768,
    )


@pytest.fixture(scope="function")
def db_engine(test_settings: Settings) -> Generator[Engine, Any, None]:
    """Yield a SQLAlchemy engine for the test database.

    Creates and drops the database schema.
    """
    engine = create_engine(test_settings.DATABASE_URL)
    dal.create_database_schema(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(db_engine: Engine) -> Generator[Session, Any, None]:
    """Yield a SQLAlchemy session for a single test.

    Rolls back transactions to ensure test isolation.
    """
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def mock_db_session() -> Generator[MagicMock, Any, None]:
    """Provide a mock of the SQLAlchemy session.

    This fixture patches the `get_session` context manager to yield a
    MagicMock object, preventing any real database connections during tests.
    """
    with patch(
        "py_name_entity_normalization.database.connection.get_session"
    ) as mock_get_session:
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session
        yield mock_session


@pytest.fixture
def mock_embedder(test_settings: Settings) -> MagicMock:
    """Provide a mock of the IEmbedder interface."""
    embedder = MagicMock(spec=IEmbedder)
    dim = test_settings.EMBEDDING_MODEL_DIMENSION
    embedder.get_model_name.return_value = "test/dummy-bert"
    embedder.get_dimension.return_value = dim
    embedder.encode.return_value = np.random.rand(dim)
    # The mock dataframe has 3 rows, so we need 3 embeddings
    embedder.encode_batch.return_value = np.random.rand(3, dim)
    return embedder


@pytest.fixture
def mock_ranker() -> MagicMock:
    """Provide a mock of the IRanker interface."""
    ranker = MagicMock(spec=IRanker)

    def dummy_rank(query: str, candidates: List[Candidate]) -> List[RankedCandidate]:
        # Simply assign a dummy score and return as RankedCandidate
        return [
            RankedCandidate(**c.model_dump(), rerank_score=0.99) for c in candidates
        ]

    ranker.rank.side_effect = dummy_rank
    return ranker


@pytest.fixture
def comprehensive_candidates() -> List[Candidate]:
    """Provide a more comprehensive list of sample Candidate objects."""
    return [
        # Exact match, different cases
        Candidate(
            concept_id=1,
            concept_name="Aspirin",
            vocabulary_id="RxNorm",
            concept_class_id="Ingredient",
            domain_id="Drug",
            distance=0.01,
        ),
        # Slight variation
        Candidate(
            concept_id=2,
            concept_name="Aspirin 81mg",
            vocabulary_id="RxNorm",
            concept_class_id="Clinical Drug",
            domain_id="Drug",
            distance=0.05,
        ),
        # Synonym
        Candidate(
            concept_id=3,
            concept_name="Acetylsalicylic Acid",
            vocabulary_id="RxNorm",
            concept_class_id="Ingredient",
            domain_id="Drug",
            distance=0.1,
        ),
        # Completely different drug
        Candidate(
            concept_id=4,
            concept_name="Ibuprofen",
            vocabulary_id="RxNorm",
            concept_class_id="Ingredient",
            domain_id="Drug",
            distance=0.7,
        ),
        # Ambiguous name - medical condition
        Candidate(
            concept_id=5,
            concept_name="Cold",
            vocabulary_id="SNOMED",
            concept_class_id="Clinical Finding",
            domain_id="Condition",
            distance=0.2,
        ),
        # Ambiguous name - drug
        Candidate(
            concept_id=6,
            concept_name="Cold and Flu Relief",
            vocabulary_id="RxNorm",
            concept_class_id="Branded Drug",
            domain_id="Drug",
            distance=0.25,
        ),
        # Very specific drug name
        Candidate(
            concept_id=7,
            concept_name="Lisinopril 20mg Tablet",
            vocabulary_id="RxNorm",
            concept_class_id="Clinical Drug Dose",
            domain_id="Drug",
            distance=0.15,
        ),
        # A non-drug concept
        Candidate(
            concept_id=8,
            concept_name="Blood Pressure",
            vocabulary_id="SNOMED",
            concept_class_id="Observable Entity",
            domain_id="Measurement",
            distance=0.9,
        ),
        # A misspelling
        Candidate(
            concept_id=9,
            concept_name="Asparin",
            vocabulary_id="RxNorm",
            concept_class_id="Ingredient",
            domain_id="Drug",
            distance=0.12,
        ),
        # Another similar drug
        Candidate(
            concept_id=10,
            concept_name="Aspirin-C",
            vocabulary_id="RxNorm",
            concept_class_id="Branded Drug",
            domain_id="Drug",
            distance=0.08,
        ),
    ]


@pytest.fixture
def mock_pandas_read_csv(mocker: MockerFixture) -> MagicMock:
    """Mock pandas.read_csv to return a controlled DataFrame iterator."""
    dummy_df = pd.DataFrame(
        {
            "concept_id": [1, 2, 3],
            "concept_name": ["Aspirin", "Tylenol", "Ibuprofen"],
            "domain_id": ["Drug", "Drug", "Drug"],
            "vocabulary_id": ["RxNorm", "RxNorm", "RxNorm"],
            "concept_class_id": ["Ingredient", "Ingredient", "Ingredient"],
        }
    )
    # Return an iterator to simulate chunking
    return mocker.patch("pandas.read_csv", return_value=[dummy_df])
