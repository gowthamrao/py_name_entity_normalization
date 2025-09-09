"""
Defines the Pydantic data models for the application.

These schemas are used for data validation, serialization, and ensuring
a consistent data structure throughout the normalization pipeline.
Using Pydantic V2 provides robustness and type safety.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class NormalizationInput(BaseModel):
    """
    Schema for the input to the normalization engine.
    """

    text: str = Field(..., description="The entity text to be normalized.")
    domains: Optional[List[str]] = Field(
        None, description="Optional list of OMOP domains to filter results."
    )


class Candidate(BaseModel):
    """
    Schema for a single candidate retrieved from the database (Stage 1).
    """

    concept_id: int = Field(..., description="The OMOP Concept ID.")
    concept_name: str = Field(..., description="The name of the concept.")
    vocabulary_id: str = Field(
        ..., description="The source vocabulary (e.g., 'RxNorm')."
    )
    concept_class_id: str = Field(
        ..., description="The concept class (e.g., 'Ingredient')."
    )
    domain_id: str = Field(..., description="The OMOP domain (e.g., 'Drug').")
    distance: float = Field(
        ...,
        description=(
            "The distance metric (e.g., cosine distance) from the query vector. "
            "Lower is better."
        ),
    )

    class Config:
        # Pydantic V2: use 'from_attributes' instead of 'orm_mode'
        from_attributes = True


class RankedCandidate(Candidate):
    """
    Schema for a candidate after re-ranking (Stage 2).
    """

    rerank_score: float = Field(
        ...,
        description="The score from the re-ranking model. Higher is better.",
    )


class NormalizationOutput(BaseModel):
    """
    Schema for the final output of the normalization engine.
    """

    input: NormalizationInput = Field(..., description="The original input query.")
    candidates: List[RankedCandidate] = Field(
        [], description="A list of ranked and sorted candidate entities."
    )
