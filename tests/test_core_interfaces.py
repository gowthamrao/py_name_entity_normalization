"""Tests for the abstract base classes in core.interfaces."""

import pytest

from py_name_entity_normalization.core.interfaces import IEmbedder, IRanker


def test_iembedder_interfaces_raise_not_implemented():
    """Test that calling IEmbedder abstract methods raises NotImplementedError."""
    # Temporarily make the class concrete for instantiation
    # by clearing the abstract methods set.
    IEmbedder.__abstractmethods__ = frozenset()

    embedder = IEmbedder()
    with pytest.raises(NotImplementedError):
        embedder.encode("test")
    with pytest.raises(NotImplementedError):
        embedder.encode_batch(["test"])
    with pytest.raises(NotImplementedError):
        embedder.get_model_name()
    with pytest.raises(NotImplementedError):
        embedder.get_dimension()


def test_iranker_interfaces_raise_not_implemented():
    """Tests that calling the abstract methods of IRanker raises NotImplementedError."""
    # Temporarily make the class concrete for instantiation
    IRanker.__abstractmethods__ = frozenset()

    ranker = IRanker()
    with pytest.raises(NotImplementedError):
        ranker.rank("query", [])
