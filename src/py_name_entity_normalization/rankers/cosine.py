"""
Baseline re-ranker based on Cosine Similarity.
"""
from typing import List

from ..core.interfaces import IRanker
from ..core.schemas import Candidate, RankedCandidate


class CosineSimilarityRanker(IRanker):
    """
    A baseline ranker that uses the cosine distance from the ANN search.

    This ranker doesn't perform any new calculations. It simply transforms the
    'distance' metric (where lower is better) from the initial search into a
    'rerank_score' (where higher is better), ensuring a consistent output format
    for the pipeline. The cosine similarity is calculated as `1 - cosine_distance`.
    """

    def rank(self, query: str, candidates: List[Candidate]) -> List[RankedCandidate]:
        """
        "Re-ranks" candidates by converting their distance to a similarity score.

        Args:
            query: The original query text (unused in this ranker).
            candidates: A list of Candidate objects from the database.

        Returns:
            A list of RankedCandidate objects, sorted by the new similarity
            score in descending order.
        """
        ranked_candidates = []
        for candidate in candidates:
            # Cosine similarity = 1 - cosine distance
            # pgvector's cosine distance is 1 - cosine_similarity.
            # So the distance is already what we need to subtract from 1.
            similarity_score = 1.0 - candidate.distance

            ranked_candidate = RankedCandidate(
                **candidate.model_dump(), rerank_score=similarity_score
            )
            ranked_candidates.append(ranked_candidate)

        # Sort by the new score in descending order (higher is better)
        ranked_candidates.sort(key=lambda x: x.rerank_score, reverse=True)

        return ranked_candidates
