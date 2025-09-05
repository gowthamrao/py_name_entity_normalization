"""
Contains the logic for building the vector index from source data.
"""
import logging

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from tqdm import tqdm

from ..config import Settings
from ..core.interfaces import IEmbedder
from ..database import dal
from ..database.connection import engine
from ..embedders.factory import get_embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Handles the creation of the database index from OMOP concept data.
    """

    def __init__(self, settings: Settings):
        """
        Initializes the IndexBuilder.

        Args:
            settings: The application settings object.
        """
        self.settings = settings
        self.embedder: IEmbedder = get_embedder(self.settings)
        # Ensure the configured embedding dimension matches the model's actual dimension
        model_dim = self.embedder.get_dimension()
        if self.settings.EMBEDDING_MODEL_DIMENSION != model_dim:
            raise ValueError(
                f"Configuration error: EMBEDDING_MODEL_DIMENSION ({self.settings.EMBEDDING_MODEL_DIMENSION}) "
                f"does not match the actual model dimension ({model_dim})."
            )

    def build_index_from_csv(
        self, csv_path: str, session: Session, force: bool = False
    ) -> None:
        """
        Builds the entire index from a CSV file of OMOP concepts.

        The process includes:
        1. Optionally dropping and recreating the database schema.
        2. Reading and processing the CSV in batches.
        3. Generating embeddings for each batch.
        4. Inserting the data into the database.
        5. Storing index metadata (e.g., embedding model name).
        6. Creating the HNSW index for fast ANN search.

        Args:
            csv_path: The file path to the OMOP CONCEPT.csv file.
            session: The SQLAlchemy session.
            force: If True, drops all existing data and schema before building.
        """
        if force:
            logger.info("Force option enabled. Dropping and recreating database schema...")
            dal.Base.metadata.drop_all(engine)
            dal.create_database_schema(engine)
            logger.info("Schema recreated.")

        logger.info(f"Reading OMOP concepts from {csv_path}...")
        # Use chunking to handle large files
        chunk_iter = pd.read_csv(
            csv_path,
            chunksize=self.settings.INDEXING_BATCH_SIZE,
            sep="\t",
            usecols=["concept_id", "concept_name", "domain_id", "vocabulary_id", "concept_class_id"],
            on_bad_lines="skip", # Some OMOP files can have parsing issues
        )

        total_rows = sum(1 for row in open(csv_path, 'r')) # Get total for tqdm
        with tqdm(total=total_rows, desc="Indexing Concepts") as pbar:
            for chunk in chunk_iter:
                chunk.dropna(subset=["concept_name"], inplace=True)
                chunk = chunk[chunk["concept_name"].str.len() > 1]
                if chunk.empty:
                    pbar.update(self.settings.INDEXING_BATCH_SIZE)
                    continue

                # Generate embeddings for the batch
                embeddings = self.embedder.encode_batch(chunk["concept_name"].tolist())
                chunk["embedding"] = list(embeddings)

                # Insert the batch into the database
                dal.bulk_insert_omop_concepts(session, chunk)
                pbar.update(len(chunk))

        logger.info("All concepts have been inserted.")
        logger.info("Storing index metadata...")

        # Store metadata about the index build
        metadata_to_store = {
            "name": self.embedder.get_model_name(),
            "dimension": self.embedder.get_dimension(),
        }
        dal.upsert_index_metadata(
            session, key="embedding_model_name", value=metadata_to_store
        )
        logger.info(f"Metadata stored: {metadata_to_store}")

        logger.info("Creating HNSW index for fast vector search. This may take a while...")
        # Create the HNSW index using the appropriate distance function
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        session.execute(
            text(
                "CREATE INDEX ON omop_concept_index USING hnsw (embedding vector_cosine_ops);"
            )
        )
        session.commit()
        logger.info("HNSW index created successfully.")
        logger.info("Index build complete.")
