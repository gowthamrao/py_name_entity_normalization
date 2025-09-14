"""Factory for creating embedder instances."""

from ..config import Settings
from ..core.interfaces import IEmbedder
from .sentence_transformer import SentenceTransformerEmbedder


def get_embedder(settings: Settings) -> IEmbedder:
    """Instantiate and return an embedder based on the application settings.

    Currently, it only supports SentenceTransformerEmbedder, but it can be
    extended to support other embedder types in the future.

    Args:
        settings: The application settings object.

    Returns:
        An instance of a class that implements the IEmbedder interface.

    """
    # In a more complex scenario, we could use a dictionary mapping
    # model types to classes. For now, we only have one type.
    return SentenceTransformerEmbedder(model_name=settings.EMBEDDING_MODEL_NAME)
