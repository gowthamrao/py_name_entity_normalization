"""Text preprocessing utilities.

This module contains functions for cleaning and preparing text before it is
fed into the embedding model. Consistent preprocessing is key to achieving
good performance.
"""

import re


def clean_text(text: str) -> str:
    """Perform basic cleaning of input text.

    The cleaning steps include:
    - Lowercasing the text.
    - Removing extra whitespace.
    - Removing special characters that are unlikely to be informative.

    Args:
    ----
        text: The input string.

    Returns:
    -------
        The cleaned string.

    """
    if not isinstance(text, str):
        return ""
    # Lowercase the text
    text = text.lower()
    # Remove special characters, keeping alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text
