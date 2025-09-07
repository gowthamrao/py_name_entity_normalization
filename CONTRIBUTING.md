# Contributing to pyNameEntityNormalization

First off, thank you for considering contributing to this project! Any contribution, whether it's a bug report, a new feature, or a documentation improvement, is greatly appreciated.

## Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management.

### Setting up the Development Environment

1.  **Fork and Clone the Repository**

    First, fork the repository to your own GitHub account. Then, clone your fork locally:

    ```bash
    git clone https://github.com/YOUR_USERNAME/pyNameEntityNormalization.git
    cd pyNameEntityNormalization
    ```

2.  **Install Dependencies**

    This project uses Poetry to manage its dependencies. To install all the required packages, including the development dependencies, run:

    ```bash
    poetry install
    ```

3.  **Set up Pre-commit Hooks**

    This project uses `pre-commit` to ensure code quality before commits. To set up the hooks, run:

    ```bash
    poetry run pre-commit install
    ```
    Now, `black` and `ruff` will run automatically on every commit.

## Running Tests and Checks

To ensure that your changes are correct and don't introduce any regressions, please run the following checks before submitting a pull request.

### Running the Test Suite

To run all the unit tests, use `pytest`:

```bash
poetry run pytest
```

### Running the Linter

To run the linter (`ruff`), use the following command:

```bash
poetry run ruff check .
```

### Running the Type Checker

To run the static type checker (`mypy`), use the following command:

```bash
poetry run mypy .
```

### Running the Format Checker

To check if the code is formatted correctly with `black`, run:

```bash
poetry run black --check .
```

## Submitting a Pull Request

1.  Create a new branch for your changes:
    ```bash
    git checkout -b my-new-feature
    ```
2.  Make your changes and commit them with a clear commit message.
3.  Push your branch to your fork:
    ```bash
    git push origin my-new-feature
    ```
4.  Open a pull request from your fork to the `main` branch of the original repository.
5.  In the pull request description, please describe the changes you made and reference any related issues.
