"""Data Access Layer (DAL) for database interactions.

This module provides a set of functions to interact with the database,
encapsulating all SQLAlchemy query logic. This separation of concerns makes
the rest of the application independent of the database schema and query details.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..core.schemas import Candidate
from .models import Base, IndexMetadata, OMOPIndex


def create_database_schema(engine) -> None:
    """Create all tables in the database based on the ORM models.

    Args:
    ----
        engine: The SQLAlchemy engine instance.

    """
    Base.metadata.create_all(engine)


def get_index_metadata(session: Session) -> Dict[str, Any]:
    """Retrieve all metadata key-value pairs from the database.

    Args:
    ----
        session: The SQLAlchemy session.

    Returns:
    -------
        A dictionary containing all metadata.

    """
    stmt = select(IndexMetadata)
    results = session.execute(stmt).scalars().all()
    return {item.key: item.value for item in results}


def upsert_index_metadata(session: Session, key: str, value: Dict[str, Any]) -> None:
    """Insert or update a metadata key-value pair.

    Args:
    ----
        session: The SQLAlchemy session.
        key: The metadata key.
        value: The metadata value (a dictionary).

    """
    stmt = (
        insert(IndexMetadata)
        .values(key=key, value=value)
        .on_conflict_do_update(index_elements=["key"], set_={"value": value})
    )
    session.execute(stmt)
    session.commit()


def bulk_insert_omop_concepts(session: Session, concepts_df: pd.DataFrame) -> None:
    """Perform a bulk insert of OMOP concepts with their embeddings.

    Args:
    ----
        session: The SQLAlchemy session.
        concepts_df: A pandas DataFrame with columns matching the OMOPIndex model.

    """
    session.bulk_insert_mappings(OMOPIndex, concepts_df.to_dict(orient="records"))
    session.commit()


def find_nearest_neighbors(
    session: Session,
    query_vector: List[float],
    k: int,
    domains: Optional[List[str]] = None,
) -> List[Candidate]:
    """Find the top-k nearest neighbors for a given query vector.

    This function uses the cosine distance operator (<=>) from pgvector for
    efficient Approximate Nearest Neighbor (ANN) search.

    Args:
    ----
        session: The SQLAlchemy session.
        query_vector: The embedding vector for the query text.
        k: The number of nearest neighbors to retrieve.
        domains: An optional list of OMOP domain_ids to filter the search.

    Returns:
    -------
        A list of Candidate objects, including their distance.

    """
    stmt = (
        select(
            OMOPIndex,
            OMOPIndex.embedding.cosine_distance(query_vector).label("distance"),
        )
        .order_by(OMOPIndex.embedding.cosine_distance(query_vector))
        .limit(k)
    )

    if domains:
        stmt = stmt.where(OMOPIndex.domain_id.in_(domains))

    results = session.execute(stmt).all()

    # The result is a list of tuples (OMOPIndex, distance)
    candidates = [
        Candidate(
            concept_id=row.OMOPIndex.concept_id,
            concept_name=row.OMOPIndex.concept_name,
            vocabulary_id=row.OMOPIndex.vocabulary_id,
            concept_class_id=row.OMOPIndex.concept_class_id,
            domain_id=row.OMOPIndex.domain_id,
            distance=row.distance,
        )
        for row in results
    ]
    return candidates
