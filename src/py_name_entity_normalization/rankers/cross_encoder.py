"""Re-ranker implementation using a Cross-Encoder model."""

from typing import Any, List, cast

try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover - provide minimal stub
    import sys
    import types

    torch = types.ModuleType("torch")
    cuda = cast(Any, types.ModuleType("cuda"))

    def _is_available() -> bool:  # pragma: no cover - runtime stub
        return False

    cuda.is_available = _is_available
    torch.cuda = cuda
    sys.modules["torch"] = torch

try:  # pragma: no cover - optional dependency
    from sentence_transformers.cross_encoder import CrossEncoder
except ModuleNotFoundError:  # pragma: no cover - allow tests to patch
    CrossEncoder = cast(Any, None)

from ..core.interfaces import IRanker
from ..core.schemas import Candidate, RankedCandidate


class CrossEncoderRanker(IRanker):
    """A re-ranker that uses a Cross-Encoder model for more accurate scoring.

    Cross-Encoders jointly process a pair of texts (e.g., query and candidate name)
    and output a single score, which is generally more accurate than using
    cosine similarity on standalone embeddings.
    """

    def __init__(self, model_name: str, device: str | None = None):
        """Initialize the CrossEncoderRanker.

        Args:
        ----
            model_name: The name of the Cross-Encoder model to load.
            device: The device to run the model on (e.g., 'cpu', 'cuda').
                    If None, it will auto-detect CUDA availability.

        """
        self._model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading cross-encoder model '{self._model_name}'...")
        print(f"Using device: '{self.device}'")
        # max_length can be important for performance
        self.model = CrossEncoder(self._model_name, max_length=512, device=self.device)
        print("Cross-encoder loaded successfully.")

    def rank(self, query: str, candidates: List[Candidate]) -> List[RankedCandidate]:
        """Re-ranks candidates using the Cross-Encoder model.

        Args:
        ----
            query: The original query text.
            candidates: A list of Candidate objects from the database.

        Returns:
        -------
            A list of RankedCandidate objects, sorted by the cross-encoder's
            score in descending order.

        """
        if not candidates:
            return []

        # Create pairs of (query, candidate_name) for the model
        sentence_pairs = [(query, candidate.concept_name) for candidate in candidates]

        # Get scores from the model
        scores = self.model.predict(sentence_pairs, convert_to_numpy=True)

        # Create RankedCandidate objects with the new scores
        ranked_candidates = [
            RankedCandidate(**candidate.model_dump(), rerank_score=score)
            for candidate, score in zip(candidates, scores)
        ]

        # Sort by the new score in descending order
        ranked_candidates.sort(key=lambda x: x.rerank_score, reverse=True)

        return ranked_candidates
