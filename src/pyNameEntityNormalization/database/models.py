"""
Defines the SQLAlchemy ORM models for the database.

This module contains the schema for the tables that store the OMOP concept
embeddings and the index metadata. The schema is defined using SQLAlchemy 2.0
declarative mapping with full type annotations.
"""
from datetime import datetime
from typing import List

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ..config import settings


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass


class IndexMetadata(Base):
    """
    Stores metadata about the generated index.

    This is crucial for runtime checks, especially to ensure that the
    embedding model used by the engine matches the one used to create
    the index.
    """

    __tablename__ = "index_metadata"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    value: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<IndexMetadata(key='{self.key}', value='{self.value}')>"


class OMOPIndex(Base):
    """
    ORM model for storing OMOP concepts and their vector embeddings.
    """

    __tablename__ = "omop_concept_index"

    concept_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    concept_name: Mapped[str] = mapped_column(String, nullable=False)
    domain_id: Mapped[str] = mapped_column(String, index=True)
    vocabulary_id: Mapped[str] = mapped_column(String, index=True)
    concept_class_id: Mapped[str] = mapped_column(String, index=True)

    # The embedding vector. The dimension must match the model used for indexing.
    embedding: Mapped[List[float]] = mapped_column(
        Vector(settings.EMBEDDING_MODEL_DIMENSION)
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<OMOPIndex(concept_id={self.concept_id}, name='{self.concept_name}')>"
