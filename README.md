# pyNameEntityNormalization

`pyNameEntityNormalization` is a production-ready Python package for high-performance Named Entity Normalization (NEN). It is designed to map biomedical entities to standard OMOP Concept IDs using a state-of-the-art hybrid "generate and rank" pipeline.

The system is built on a robust and scalable architecture, leveraging a PostgreSQL backend with the `pgvector` extension for efficient similarity search and `sentence-transformers` for cutting-edge text embeddings.

## Key Features

- **Scalable Backend**: Utilizes PostgreSQL and `pgvector` for storing millions of concept embeddings and performing efficient Approximate Nearest Neighbor (ANN) searches.
- **Hybrid Search Pipeline**:
    - **Stage 1 (Generate)**: A fast ANN search retrieves the Top-K most likely candidates from the database.
    - **Stage 2 (Rank)**: A configurable, more powerful model (e.g., a Cross-Encoder) re-ranks the candidates for higher accuracy.
- **Model Agnostic**: Easily configurable to use different embedding models from the `sentence-transformers` library (e.g., SapBERT, GTE).
- **Configurable Ranking**: Switch between ranking strategies (`Cosine Similarity`, `Cross-Encoder`, or a placeholder for an `LLM`) via a simple configuration setting.
- **Model Consistency Check**: Guarantees that the model being used at runtime is the same one that was used to build the index, preventing invalid results.
- **Modern & Robust**: Built with Python 3.10+, fully type-hinted, and based on industry-standard libraries like Pydantic V2, SQLAlchemy 2.0, and Typer.

## Installation

This project is managed with [Poetry](https://python-poetry.org/).

To install the package and its dependencies, run:

```bash
poetry install
```

This will create a virtual environment and install all the necessary packages.

For instructions on how to set up a development environment, please see our [Contributing Guide](./CONTRIBUTING.md).

## Configuration

The application is configured via environment variables. You can create a `.env` file in your project root or set the variables directly.

| Variable                      | Description                                                               | Default                                           |
| ----------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------- |
| `DATABASE_URL`                | SQLAlchemy connection string for your PostgreSQL DB.                      | `postgresql+psycopg://user:password@localhost:5432/nen_db` |
| `EMBEDDING_MODEL_NAME`        | The `sentence-transformers` model for embeddings.                         | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`     |
| `EMBEDDING_MODEL_DIMENSION`   | The vector dimension of the embedding model.                              | `768`                                             |
| `RERANKING_STRATEGY`          | The re-ranking strategy (`cosine`, `cross_encoder`, `llm`).                 | `cosine`                                          |
| `CROSS_ENCODER_MODEL_NAME`    | The `sentence-transformers` model for the cross-encoder ranker.           | `cross-encoder/ms-marco-MiniLM-L-6-v2`            |
| `DEFAULT_TOP_K`               | The default number of candidates to retrieve in Stage 1.                  | `50`                                              |
| `DEFAULT_CONFIDENCE_THRESHOLD`| The default similarity score threshold to filter candidates.            | `0.85`                                            |


## Usage

Using the package involves two main steps: building the index and then using the engine to normalize entities.

For a detailed, end-to-end example of how to integrate this package with a Named Entity Recognition (NER) tool, see our **[How to Use Guide](./docs/how_to_use_guide.md)**.

### 1. Building the Index (CLI)

The package provides a CLI, `nen-indexer`, to build the vector search index from an OMOP `CONCEPT.csv` file.

First, ensure your PostgreSQL database is running and the `pgvector` extension is enabled (`CREATE EXTENSION vector;`).

Then, run the `build-index` command:

```bash
# Get help
nen-indexer --help

# Build the index from your CONCEPT.csv file
# Note: The default OMOP CONCEPT.csv is tab-separated
nen-indexer build-index /path/to/your/CONCEPT.csv

# Force a rebuild, dropping all existing data
nen-indexer build-index /path/to/your/CONCEPT.csv --force
```

You can also verify that the index was built correctly and is compatible with your current configuration:

```bash
nen-indexer verify-index
```

### 2. Normalizing Entities (Python Library)

Once the index is built, you can use the `NormalizationEngine` in your Python code.

```python
import os
from pyNameEntityNormalization import NormalizationEngine, NormalizationInput, Settings

# It's good practice to manage settings explicitly
# This assumes your .env file is configured
# or environment variables are set.
my_settings = Settings()

# The engine will perform a model consistency check on initialization
try:
    engine = NormalizationEngine(settings=my_settings)

    # Create an input object
    # You can optionally filter by OMOP domain
    norm_input = NormalizationInput(
        text="aspirin 81 mg oral tablet",
        domains=["Drug"]
    )

    # Normalize the entity
    result = engine.normalize(norm_input)

    # Print the results
    print(f"Input: '{result.input.text}'")
    for candidate in result.candidates:
        print(
            f"  - Concept ID: {candidate.concept_id:<10} "
            f"| Name: {candidate.concept_name:<40} "
            f"| Score: {candidate.rerank_score:.4f}"
        )

except ValueError as e:
    print(f"An error occurred: {e}")
    # This can happen if the model in your settings doesn't match the
    # model used to build the index.

except Exception as e:
    print(f"Could not initialize the engine: {e}")
    # This can happen if the database is not available or not yet indexed.
```

## Architecture Overview

The package is designed with modularity in mind, using interfaces and factories to decouple components.

- **`core/engine.py`**: The `NormalizationEngine` orchestrates the entire process.
- **`core/interfaces.py`**: Defines the `IEmbedder` and `IRanker` abstract base classes.
- **`database/`**: Contains all SQLAlchemy models (`models.py`), connection logic (`connection.py`), and the Data Access Layer (`dal.py`).
- **`embedders/`**: Contains the `SentenceTransformerEmbedder` for generating embeddings.
- **`rankers/`**: Contains the different re-ranking strategies (`CosineSimilarityRanker`, `CrossEncoderRanker`).
- **`indexer/`**: Contains the `IndexBuilder` logic for the offline indexing process.
- **`cli.py`**: The Typer-based command-line interface.
