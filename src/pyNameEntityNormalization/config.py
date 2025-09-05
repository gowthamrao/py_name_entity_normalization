"""
Configuration management for the application.

This module uses pydantic-settings to load configuration from environment
variables, providing a single, type-safe source of truth for all settings.
This approach makes the application easily configurable in different
environments (development, testing, production).
"""
from typing import Dict, List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # Pydantic V2 model_config, replaces Config class
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Database Configuration ---
    DATABASE_URL: str = Field(
        "postgresql+psycopg2://user:password@localhost:5432/nen_db",
        description="SQLAlchemy database connection string.",
    )

    # --- Embedding Model Configuration ---
    EMBEDDING_MODEL_NAME: str = Field(
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        description="The name of the sentence-transformer model to use for embeddings.",
    )
    EMBEDDING_MODEL_DIMENSION: int = Field(
        768,
        description="The dimension of the embeddings produced by the model. SapBERT is 768.",
    )

    # --- Ranker Configuration ---
    RERANKING_STRATEGY: str = Field(
        "cosine",
        description="The re-ranking strategy to use ('cosine', 'cross_encoder', 'llm').",
    )
    CROSS_ENCODER_MODEL_NAME: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="The name of the cross-encoder model for re-ranking.",
    )

    # --- Normalization Engine Configuration ---
    DEFAULT_TOP_K: int = Field(
        50, description="The default number of candidates to retrieve from ANN search."
    )
    DEFAULT_CONFIDENCE_THRESHOLD: float = Field(
        0.85,
        description="Default cosine similarity threshold to filter candidates before re-ranking.",
    )
    # Example: map 'DISEASE' from a NER tool to OMOP 'Condition' domain
    NER_TO_OMOP_DOMAIN_MAPPING: Dict[str, str] = Field(
        default_factory=lambda: {"DISEASE": "Condition", "DRUG": "Drug"},
        description="Mapping from NER entity labels to OMOP domain IDs.",
    )

    # --- Indexing Configuration ---
    INDEXING_BATCH_SIZE: int = Field(
        1024, description="The batch size for generating and inserting embeddings during indexing."
    )


# Create a single, importable instance of the settings
settings = Settings()
