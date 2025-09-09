"""
Command-Line Interface (CLI) for the py_name_entity_normalization package.

This module provides commands for managing the normalization index, such as
building it from source data and verifying its status. It uses the Typer
library to create a user-friendly and well-documented CLI.
"""
import logging
from pathlib import Path

import typer
from rich.console import Console
from typing_extensions import Annotated

from .config import Settings
from .core.engine import NormalizationEngine
from .database.connection import get_session
from .indexer.builder import IndexBuilder

# Initialize Typer app and Rich console for nice output
app = typer.Typer(
    help="CLI for building and managing the Named Entity Normalization index."
)
console = Console()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.command()
def build_index(
    csv_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the OMOP CONCEPT.csv file.",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Drop and recreate the entire database schema before indexing.",
        ),
    ] = False,
):
    """
    Builds the vector index from a CSV file of OMOP concepts.
    """
    console.rule("[bold green]Starting Index Build[/bold green]")
    try:
        settings = Settings()
        index_builder = IndexBuilder(settings)
        with get_session() as session:
            index_builder.build_index_from_csv(
                csv_path=str(csv_path), session=session, force=force
            )
        console.rule("[bold green]Index Build Successful[/bold green]")
        console.print("✅ Index has been built and is ready for use.")
    except Exception as e:
        console.print(f"[bold red]Error during index build:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def verify_index():
    """
    Verifies the status and metadata of the existing index.
    """
    console.rule("[bold blue]Verifying Index[/bold blue]")
    try:
        settings = Settings()
        # Initializing the engine performs the consistency check
        engine = NormalizationEngine(settings)
        console.print("✅ NormalizationEngine initialized successfully.")
        console.print("✅ Model consistency check passed.")
        console.print(
            f"   [dim]Current Model:[/dim] [bold]{engine.embedder.get_model_name()}[/bold]"
        )

        # Further verification can be added here, e.g., count items
        # with get_session() as session:
        #   count = session.query(OMOPIndex).count()
        #   console.print(f"   [dim]Indexed Concepts:[/dim] [bold]{count}[/bold]")

        console.rule("[bold blue]Verification Complete[/bold blue]")
    except Exception as e:
        console.print(f"[bold red]Error during index verification:[/bold red] {e}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
