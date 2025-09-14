"""The main NormalizationEngine orchestrating the entire pipeline."""

import logging
from typing import List, Optional

from ..config import Settings
from ..config import settings as default_settings
from ..core.interfaces import IEmbedder, IRanker
from ..core.schemas import NormalizationInput, NormalizationOutput, RankedCandidate
from ..database import dal
from ..database.connection import get_session
from ..embedders.factory import get_embedder
from ..rankers.factory import get_ranker
from ..utils.preprocessing import clean_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NormalizationEngine:
    """The main orchestrator for the named entity normalization pipeline.

    This class integrates all the components: configuration, database access,
    embedding generation, and re-ranking to provide a seamless normalization
    service.
    """

    def __init__(self, settings: Settings):
        """Initialize the NormalizationEngine.

        This involves setting up the embedder and ranker from factories and
        performing a crucial model consistency check.

        Args:
        ----
            settings: The application settings object.

        Raises:
        ------
            ValueError: If the embedding model used for the index does not
                        match the currently configured model.

        """
        self.settings = settings
        logger.info("Initializing NormalizationEngine...")

        # Dependency Injection: Get components from factories
        self.embedder: IEmbedder = get_embedder(self.settings)
        self.ranker: IRanker = get_ranker(self.settings)

        # --- Model Consistency Check ---
        self._verify_model_consistency()

        logger.info("NormalizationEngine initialized successfully.")

    def _verify_model_consistency(self) -> None:
        """Check if the loaded model matches the one used to build the index."""
        logger.info("Performing model consistency check...")
        with get_session() as session:
            metadata = dal.get_index_metadata(session)
            indexed_model_name = metadata.get("embedding_model_name", {}).get("name")

        if not indexed_model_name:
            logger.warning(
                "Index metadata not found or model name is missing. "
                "Skipping model consistency check. "
                "This is expected if the index hasn't been built yet."
            )
            return

        current_model_name = self.embedder.get_model_name()
        if current_model_name != indexed_model_name:
            error_message = (
                "Model mismatch! The current model "
                f"('{current_model_name}') is different from the model "
                f"used to build the index ('{indexed_model_name}')."
            )
            raise ValueError(error_message)
        logger.info("Model consistency check passed.")

    def normalize(
        self,
        input_data: NormalizationInput,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> NormalizationOutput:
        """Execute the full normalization pipeline for a given input.

        Pipeline: Preprocess -> Embed -> Stage 1 (Generate) -> Threshold ->
        Stage 2 (Rank)

        Args:
        ----
            input_data: The input data containing the text to normalize.
            top_k: The number of candidates to retrieve from the ANN search.
                   Overrides the default if provided.
            threshold: The similarity threshold to filter candidates.
                       Overrides the default if provided.

        Returns:
        -------
            An object containing the original input and the list of ranked candidates.

        """
        # Set defaults from settings if not provided
        k = top_k or self.settings.DEFAULT_TOP_K
        conf_threshold = threshold or self.settings.DEFAULT_CONFIDENCE_THRESHOLD

        # 1. Preprocess text
        processed_text = clean_text(input_data.text)
        if not processed_text:
            return NormalizationOutput(input=input_data, candidates=[])

        # 2. Embed the query text
        query_vector = self.embedder.encode(processed_text).tolist()

        # 3. Stage 1: Generate candidates from database
        with get_session() as session:
            ann_candidates = dal.find_nearest_neighbors(
                session, query_vector, k, input_data.domains
            )

        # 4. Threshold Check (convert distance to similarity)
        # Cosine similarity = 1 - Cosine Distance
        filtered_candidates = [
            c for c in ann_candidates if (1.0 - c.distance) >= conf_threshold
        ]

        if not filtered_candidates:
            return NormalizationOutput(input=input_data, candidates=[])

        # 5. Stage 2: Re-rank the filtered candidates
        ranked_candidates: List[RankedCandidate] = self.ranker.rank(
            processed_text, filtered_candidates
        )

        # 6. Format and return output
        return NormalizationOutput(input=input_data, candidates=ranked_candidates)


# A default, importable instance of the engine for convenience
engine: Optional[NormalizationEngine]
try:
    engine = NormalizationEngine(settings=default_settings)
except (ValueError, Exception) as e:
    logger.error(f"Failed to initialize the default NormalizationEngine: {e}")
    logger.error(
        "This may be expected if the database is not yet available or indexed."
    )
    engine = None
