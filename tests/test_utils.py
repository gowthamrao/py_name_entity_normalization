"""Tests for utility functions."""

import pytest

from py_name_entity_normalization.utils.preprocessing import clean_text


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Aspirin 100mg", "aspirin 100mg"),
        ("  Multiple   Spaces  ", "multiple spaces"),
        ("With-Special_Chars!", "with-specialchars"),
        (" Tylenol® ", "tylenol"),  # Check ® removal
        ("", ""),
        ("  ", ""),
    ],
)
def test_clean_text_happy_path(input_text: str, expected_output: str) -> None:
    """Tests various successful text cleaning scenarios."""
    assert clean_text(input_text) == expected_output


def test_clean_text_non_string_input() -> None:
    """Tests that non-string input is handled gracefully."""
    assert clean_text(None) == ""  # type: ignore
    assert clean_text(123) == ""  # type: ignore
    assert clean_text(["list"]) == ""  # type: ignore
