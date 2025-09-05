# Detailed Guide: Creating the Vector Database for pyNameEntityNormalization

This guide provides a comprehensive walkthrough of how to set up the PostgreSQL vector database required by the `pyNameEntityNormalization` package. The process involves installing `pgvector`, sourcing medical vocabularies, handling licensed vocabularies like CPT4, and finally, loading and indexing the data.

## Table of Contents
1.  [Prerequisites](#prerequisites)
2.  [Step 1: Install and Configure PostgreSQL with pgvector](#step-1-install-and-configure-postgresql-with-pgvector)
3.  [Step 2: Source the OMOP Vocabulary Files](#step-2-source-the-omop-vocabulary-files)
4.  [Step 3: CPT4 Expansion (Optional but Recommended)](#step-3-cpt4-expansion-optional-but-recommended)
5.  [Step 4: Understand the Database Schema (DDL)](#step-4-understand-the-database-schema-ddl)
6.  [Step 5: Load and Index the Data](#step-5-load-and-index-the-data)

---

### Prerequisites

-   **PostgreSQL**: A running instance of PostgreSQL (version 13 or newer).
-   **Python**: Python 3.10+ with the `pyNameEntityNormalization` package installed (`pip install .`).
-   **UMLS License (Optional)**: If you need to include CPT4 codes, you must have a valid license for the UMLS Metathesaurus.

---

### Step 1: Install and Configure PostgreSQL with pgvector

`pgvector` is a PostgreSQL extension for vector similarity search. It needs to be installed on your PostgreSQL server before you can create the vector index.

#### Installation

The installation method depends on your operating system and environment.

-   **Linux (from source):**
    ```bash
    # Ensure you have postgresql development packages installed (e.g., postgresql-server-dev-15)
    git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git
    cd pgvector
    make
    sudo make install
    ```

-   **Mac (using Homebrew):**
    ```bash
    brew install pgvector
    ```

-   **Windows:**
    1.  Ensure you have the "C++ build tools" workload installed in Visual Studio.
    2.  Run the "x64 Native Tools Command Prompt" as an administrator.
    3.  Execute the following commands:
        ```cmd
        set "PGROOT=C:\Program Files\PostgreSQL\16" # Adjust to your PostgreSQL path
        cd %TEMP%
        git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git
        cd pgvector
        nmake /F Makefile.win
        nmake /F Makefile.win install
        ```

-   **Docker:**
    Many Docker images for PostgreSQL come with `pgvector` pre-installed. A popular choice is the official `pgvector/pgvector` image on Docker Hub.

    ```yaml
    # Example docker-compose.yml
    services:
      db:
        image: pgvector/pgvector:pg16
        environment:
          POSTGRES_DB: nen_db
          POSTGRES_USER: user
          POSTGRES_PASSWORD: password
        ports:
          - "5432:5432"
    ```

#### Enable the Extension

After installing `pgvector`, you must enable it in the database you intend to use. Connect to your PostgreSQL database (e.g., using `psql`) and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The `nen-indexer` tool also runs this command, but it's good practice to ensure it's enabled beforehand.

---

### Step 2: Source the OMOP Vocabulary Files

The `pyNameEntityNormalization` tool is designed to work with the standardized vocabularies from the Observational Medical Outcomes Partnership (OMOP).

1.  **Navigate to Athena:** Go to the [OHDSI Athena](https://athena.ohdsi.org/) website.
2.  **Register/Login:** You will need to create a free account to download the vocabulary files.
3.  **Select Vocabularies:** Choose the vocabularies you need. For general-purpose biomedical normalization, a good starting set includes:
    -   `SNOMED`
    -   `RxNorm`
    -   `LOINC`
    -   `CPT4` (see next section)
    -   And any other domains relevant to your use case.
4.  **Download:** Add the selected vocabularies to your bundle and click "Download". You will receive an email with a link to a zip file.
5.  **Unzip:** Download and unzip the file. Inside, you will find several `.csv` files, including `CONCEPT.csv`, which is the primary file used by the indexer.

---

### Step 3: CPT4 Expansion (Optional but Recommended)

The Current Procedural Terminology, Fourth Edition (CPT4) is a proprietary medical code set licensed by the American Medical Association (AMA). Due to this licensing, the CPT4 codes are not included directly in the Athena download. Instead, Athena provides a tool to add them if you have the appropriate license.

1.  **UMLS License:** To get CPT4 codes, you need a UMLS Metathesaurus License from the National Library of Medicine.
2.  **Locate the CPT4 Tool:** In the unzipped vocabulary folder from Athena, you will find a Java archive named `cpt4.jar` (or similar) and instructions.
3.  **Run the Tool:** This tool typically requires your UMLS API key. You would run it from the command line:

    ```bash
    java -jar cpt4.jar -U YOUR_UMLS_API_KEY
    ```
    *(Note: The exact command may vary. Please consult the instructions provided with the download.)*

    This process will download the CPT4 concepts and add them to your `CONCEPT.csv` and other relevant vocabulary files, making them available for indexing.

---

### Step 4: Understand the Database Schema (DDL)

The `pyNameEntityNormalization` package creates two tables in your database, defined in `src/pyNameEntityNormalization/database/models.py`.

#### `omop_concept_index`

This is the main table that stores the concepts and their vector embeddings.

-   `concept_id` (INTEGER, PRIMARY KEY): The OMOP concept ID.
-   `concept_name` (VARCHAR): The name of the concept (e.g., "Aspirin").
-   `domain_id` (VARCHAR): The OMOP domain (e.g., "Drug").
-   `vocabulary_id` (VARCHAR): The source vocabulary (e.g., "RxNorm").
-   `concept_class_id` (VARCHAR): The concept class.
-   `embedding` (VECTOR): The dense vector embedding of the `concept_name`.

The DDL for creating this table is approximately:
```sql
CREATE TABLE omop_concept_index (
    concept_id INTEGER NOT NULL,
    concept_name VARCHAR NOT NULL,
    domain_id VARCHAR,
    vocabulary_id VARCHAR,
    concept_class_id VARCHAR,
    embedding vector(768), -- Dimension depends on the embedding model
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    PRIMARY KEY (concept_id)
);
```

#### `index_metadata`

This table stores metadata about the index, which is critical for ensuring consistency at runtime.

-   `key` (VARCHAR, UNIQUE): The metadata key (e.g., "embedding_model_name").
-   `value` (JSON): The metadata value (e.g., `{"name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "dimension": 768}`).

---

### Step 5: Load and Index the Data

Once your database is set up and you have your `CONCEPT.csv` file (with CPT4 codes expanded, if desired), you can build the index using the provided CLI tool, `nen-indexer`.

#### The `build-index` Command

This command orchestrates the entire indexing process:

```bash
nen-indexer build-index /path/to/your/CONCEPT.csv
```

If you need to rebuild the index from scratch, you can use the `--force` flag, which will drop all existing data and tables first.

```bash
nen-indexer build-index /path/to/your/CONCEPT.csv --force
```

#### What Happens During Indexing

The `build-index` command performs the following steps:
1.  **Connects** to the PostgreSQL database specified by your `DATABASE_URL`.
2.  **Reads the `CONCEPT.csv`** file in batches to efficiently handle large vocabularies.
3.  **Generates Embeddings:** For each batch of concept names, it uses the configured sentence-transformer model (e.g., `SapBERT`) to create vector embeddings.
4.  **Bulk Inserts Data:** It inserts the concept data and their corresponding embeddings into the `omop_concept_index` table.
5.  **Stores Metadata:** It records the name and dimension of the embedding model in the `index_metadata` table. This is used later to prevent running the engine with a mismatched model.
6.  **Creates the HNSW Index:** Finally, it creates a Hierarchical Navigable Small World (HNSW) index on the `embedding` column. This is what enables the fast Approximate Nearest Neighbor (ANN) search. The command executed is:
    ```sql
    CREATE INDEX ON omop_concept_index USING hnsw (embedding vector_cosine_ops);
    ```

After the process completes, your vector database is ready for use with the `NormalizationEngine`. You can verify the index at any time by running `nen-indexer verify-index`.
