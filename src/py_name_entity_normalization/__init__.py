"""py_name_entity_normalization.

A production-ready Python package for Named Entity Normalization (NEN)
using a hybrid generate-and-rank approach with pgvector and sentence-transformers.

The package's top-level module exposes commonly used classes while keeping
heavy dependencies lazy-loaded. Importing ``py_name_entity_normalization``
should therefore be lightweight and not trigger model downloads or database
connections. Heavy modules such as :mod:`core.engine` are imported on demand
via ``__getattr__``.
"""

__version__ = "0.1.0"

from typing import Any

from .config import Settings
from .core.schemas import (
    Candidate,
    NormalizationInput,
    NormalizationOutput,
    RankedCandidate,
)


def __getattr__(name: str) -> Any:  # pragma: no cover - thin wrapper
    """Lazily import heavy modules on first access.

    This keeps ``import py_name_entity_normalization`` light-weight while still
    exposing ``NormalizationEngine`` and ``engine`` as expected.
    """
    if name in {"NormalizationEngine", "engine"}:
        from .core.engine import NormalizationEngine, engine

        return {"NormalizationEngine": NormalizationEngine, "engine": engine}[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Define what is available for public import
__all__ = [
    "NormalizationEngine",
    "engine",
    "NormalizationInput",
    "NormalizationOutput",
    "RankedCandidate",
    "Candidate",
    "Settings",
]
