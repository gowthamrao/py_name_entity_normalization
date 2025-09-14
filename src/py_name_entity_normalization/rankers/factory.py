"""Factory for creating ranker instances."""

from ..config import Settings
from ..core.interfaces import IRanker
from .cosine import CosineSimilarityRanker
from .cross_encoder import CrossEncoderRanker
from .llm import LLMRanker


def get_ranker(settings: Settings) -> IRanker:
    """Instantiate and return a ranker based on the application settings.

    This factory reads the `RERANKING_STRATEGY` from the settings and
    returns the corresponding ranker instance.

    Args:
        settings: The application settings object.

    Returns:
        An instance of a class that implements the IRanker interface.

    Raises:
        ValueError: If the configured reranking strategy is not supported.

    """
    strategy = settings.RERANKING_STRATEGY.lower()
    print(f"Initializing re-ranking strategy: '{strategy}'")

    if strategy == "cosine":
        return CosineSimilarityRanker()
    elif strategy == "cross_encoder":
        return CrossEncoderRanker(model_name=settings.CROSS_ENCODER_MODEL_NAME)
    elif strategy == "llm":
        return LLMRanker()
    else:
        raise ValueError(f"Unknown re-ranking strategy: '{strategy}'")
