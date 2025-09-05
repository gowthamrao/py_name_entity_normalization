"""
pyNameEntityNormalization

A production-ready Python package for Named Entity Normalization (NEN)
using a hybrid generate-and-rank approach with pgvector and sentence-transformers.
"""

__version__ = "0.1.0"

from .config import Settings
from .core.engine import NormalizationEngine, engine
from .core.schemas import (
    Candidate,
    NormalizationInput,
    NormalizationOutput,
    RankedCandidate,
)

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
