# How to Use This Package: A Detailed Guide

This guide provides a step-by-step walkthrough on how to use `pyNameEntityNormalization` in a complete, end-to-end workflow. We will start with raw, unstructured text, use a Named Entity Recognition (NER) tool to identify relevant entities, and then use this package to normalize those entities to standard OMOP Concept IDs.

For this guide, we will imagine you are using the `pyNameEntityRecognition` package, a hypothetical tool designed to extract biomedical entities from text.

## Table ofContents
1. [Prerequisites](#prerequisites)
2. [The End-to-End Workflow](#the-end-to-end-workflow)
3. [Code Example: From Text to Normalized Concepts](#code-example-from-text-to-normalized-concepts)
4. [Advanced Usage and Configuration](#advanced-usage-and-configuration)

---

### Prerequisites

Before you begin, ensure you have the following set up:

1.  **A Functional Vector Database**: You must have a PostgreSQL database with the `pgvector` extension enabled and loaded with OMOP concept data. If you have not done this yet, please follow our **[Detailed Guide: Creating the Vector Database](./vector_db_creation_guide.md)** first.

2.  **`pyNameEntityNormalization` Installed**: This package should be installed in your Python environment.
    ```bash
    # Install from the root of the project
    pip install .
    ```

3.  **`pyNameEntityRecognition` Installed**: For this workflow, we'll use a hypothetical NER package. Install it via pip:
    ```bash
    pip install pyNameEntityRecognition
    ```

4.  **Environment Variables Configured**: Ensure your `.env` file is configured with your database URL and model preferences as described in the `README.md`.

---

### The End-to-End Workflow

The process of normalizing entities from a block of text is a two-stage pipeline:

1.  **Named Entity Recognition (NER)**: First, we process the unstructured text to identify and extract spans of text that represent specific concepts (e.g., "aspirin 81 mg oral tablet", "diabetes"). The NER tool also provides a label for each entity (e.g., `Drug`, `Condition`).

2.  **Named Entity Normalization (NEN)**: Next, we take the extracted entity text and its label and pass it to `pyNameEntityNormalization`. The package searches the vector database for the most likely OMOP Concept ID, using the entity label to narrow the search to the correct domain (e.g., searching for "aspirin" only within the "Drug" domain).

This separation of concerns allows each component to be specialized for its task, leading to a more accurate and flexible system.

---

### Code Example: From Text to Normalized Concepts

Here is a complete, commented Python script demonstrating the workflow.

```python
import os
from pyNameEntityNormalization import NormalizationEngine, NormalizationInput, Settings

# For this example, we will mock the output of the hypothetical
# pyNameEntityRecognition package. In a real application, you would
# import and call it directly.

def mock_pyNameEntityRecognition(text: str) -> list[dict]:
    """
    A mock function that simulates the output of an NER tool.
    It takes a sentence and returns a list of found entities.
    """
    print(f"--- Running NER on: '{text}' ---")
    if "aspirin" in text and "headache" in text:
        return [
            {"text": "aspirin 81 mg oral tablet", "label": "Drug"},
            {"text": "headache", "label": "Condition"},
        ]
    return []

# 1. Your input text
raw_text = "The patient was prescribed aspirin 81 mg oral tablet for their headache."

# 2. Run Named Entity Recognition (NER)
# In a real scenario:
# from pyNameEntityRecognition import ner
# recognized_entities = ner(raw_text)
recognized_entities = mock_pyNameEntityRecognition(raw_text)

print(f"Found {len(recognized_entities)} entities: {[e['text'] for e in recognized_entities]}\n")


# 3. Initialize the Normalization Engine
# The engine connects to the database and performs a model consistency check.
try:
    print("--- Initializing Normalization Engine ---")
    # It's good practice to manage settings explicitly
    my_settings = Settings()
    engine = NormalizationEngine(settings=my_settings)
    print("Engine initialized successfully.\n")

    # 4. Loop through entities and normalize each one
    for entity in recognized_entities:
        entity_text = entity["text"]
        entity_domain = entity["label"] # e.g., "Drug", "Condition"

        print(f"--- Normalizing Entity: '{entity_text}' (Domain: {entity_domain}) ---")

        # Create an input object, using the NER label to filter by domain
        norm_input = NormalizationInput(
            text=entity_text,
            domains=[entity_domain] # Filtering by domain is highly recommended
        )

        # Normalize the entity
        result = engine.normalize(norm_input)

        # Print the top candidate
        if result.candidates:
            top_candidate = result.candidates[0]
            print(
                f"  Input: '{result.input.text}'\n"
                f"  Top Match: '{top_candidate.concept_name}' (ID: {top_candidate.concept_id})\n"
                f"  Score: {top_candidate.rerank_score:.4f}\n"
            )
        else:
            print(f"  No suitable candidates found for '{entity_text}'.\n")

except ValueError as e:
    print(f"An error occurred during normalization: {e}")
except Exception as e:
    print(f"Could not initialize the engine: {e}")
    print("Please ensure your database is running and the index has been built.")

```

#### Expected Output

Running the script above would produce an output similar to this:

```
--- Running NER on: 'The patient was prescribed aspirin 81 mg oral tablet for their headache.' ---
Found 2 entities: ['aspirin 81 mg oral tablet', 'headache']

--- Initializing Normalization Engine ---
Engine initialized successfully.

--- Normalizing Entity: 'aspirin 81 mg oral tablet' (Domain: Drug) ---
  Input: 'aspirin 81 mg oral tablet'
  Top Match: 'Aspirin 81 MG Oral Tablet' (ID: 12345)
  Score: 0.9876

--- Normalizing Entity: 'headache' (Domain: Condition) ---
  Input: 'headache'
  Top Match: 'Headache' (ID: 54321)
  Score: 0.9543

```

---

### Advanced Usage and Configuration

While the example above covers the core workflow, `pyNameEntityNormalization` offers more control over the process:

-   **Reranking Strategy**: You can change the reranking model (e.g., from `cosine` to a more powerful `cross_encoder`) via the `RERANKING_STRATEGY` environment variable. The Cross-Encoder is slower but typically more accurate.
-   **Confidence Thresholds**: The `DEFAULT_CONFIDENCE_THRESHOLD` setting allows you to filter out candidates with a similarity score below a certain value, helping to reduce false positives.
-   **Top-K Candidates**: You can adjust how many potential candidates are retrieved from the database in the initial search step by setting `DEFAULT_TOP_K`.

For a full list of configuration options, please refer to the main `README.md` file. By combining a high-quality NER model with the configurable pipeline of this package, you can build a powerful and accurate entity normalization system.
