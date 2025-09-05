"""
Placeholder for a re-ranker using a Large Language Model (LLM).
"""
from typing import List

from ..core.interfaces import IRanker
from ..core.schemas import Candidate, RankedCandidate


class LLMRanker(IRanker):
    """
    A placeholder for a future re-ranking implementation using an LLM.

    This class provides a clear extension point in the architecture for
    integrating with LLM-based re-ranking services or models.
    """

    def rank(
        self, query: str, candidates: List[Candidate]
    ) -> List[RankedCandidate]:
        """
        This method is not yet implemented and will raise an error if called.
        """
        raise NotImplementedError(
            "LLM-based re-ranking is not yet implemented."
        )
