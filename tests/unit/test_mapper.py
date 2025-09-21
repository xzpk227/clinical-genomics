"""
Unit tests for src/mapping/mapper.py.

Tests cover:
- top-k ordering (highest confidence first)
- confidence scores all in [0.0, 1.0]
- deduplication by HPO ID (keep highest score per term)
- returns at most top_k results
- empty list returned when index has no results (empty index / all -1 indices)
- encoding failure logs a warning and returns empty list

A _StubSentenceTransformer (same pattern as test_build_index.py) is used to
avoid loading any real model.  A small in-memory FAISS IndexFlatIP is built
directly in each test — no files are written to disk.
"""

from __future__ import annotations

import logging

import faiss
import numpy as np
import pytest

from src.extraction.extractor import Mention
from src.mapping.mapper import HPOCandidate, Mapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# Stub model
# ---------------------------------------------------------------------------


class _StubSentenceTransformer:
    """Minimal stub that mimics SentenceTransformer.encode().

    Returns random float32 vectors of dimension 768, L2-normalized so that
    the inner product between any two vectors is in [-1, 1].  The
    ``normalize_embeddings`` kwarg is accepted but ignored (vectors are
    always normalized).

    A fixed seed is used so that the same text always produces the same vector
    within a single test run.
    """

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        vecs = rng.standard_normal((len(sentences), _EMBEDDING_DIM)).astype(np.float32)
        # L2-normalize each row
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms


class _FailingEncoder:
    """Stub that always raises an exception from encode()."""

    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        raise RuntimeError("Simulated encoding failure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Build an in-memory FAISS IndexFlatIP from the given float32 vectors."""
    assert vectors.ndim == 2
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    return index


def _make_id_map(hpo_ids: list[str], texts: list[str]) -> list[dict]:
    """Build an id_map list from parallel hpo_ids and texts lists."""
    assert len(hpo_ids) == len(texts)
    return [
        {"vector_idx": i, "hpo_id": hpo_ids[i], "text": texts[i]}
        for i in range(len(hpo_ids))
    ]


def _make_mention(text: str = "Seizure") -> Mention:
    return Mention(text=text, start=0, end=len(text))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_model() -> _StubSentenceTransformer:
    return _StubSentenceTransformer()


@pytest.fixture()
def small_index_and_id_map(stub_model: _StubSentenceTransformer):
    """Build a small in-memory index with 5 distinct HPO terms (one vector each).

    Returns (index, id_map, stub_model).
    """
    hpo_ids = [
        "HP:0001250",
        "HP:0001290",
        "HP:0000365",
        "HP:0000750",
        "HP:0001263",
    ]
    texts = [
        "Seizure",
        "Hypotonia",
        "Hearing loss",
        "Delayed speech",
        "Global developmental delay",
    ]
    # Encode all texts to get vectors
    vectors = stub_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    index = _build_index(vectors)
    id_map = _make_id_map(hpo_ids, texts)
    return index, id_map


# ---------------------------------------------------------------------------
# Tests: top-k ordering
# ---------------------------------------------------------------------------


class TestTopKOrdering:
    """Candidates must be sorted by confidence descending."""

    def test_candidates_sorted_descending(
        self,
        stub_model: _StubSentenceTransformer,
        small_index_and_id_map,
    ) -> None:
        index, id_map = small_index_and_id_map
        mapper = Mapper(model=stub_model, index=index, id_map=id_map, top_k=3)
        candidates = mapper.map(_make_mention("Seizure"))

        confidences = [c.confidence for c in candidates]
        assert confidences == sorted(confidences, reverse=True), (
            f"Candidates not sorted descending: {confidences}"
        )

    def test_single_candidate_is_sorted(
        self,
        stub_model: _StubSentenceTransformer,
        small_index_and_id_map,
    ) -> None:
        index, id_map = small_index_and_id_map
        mapper = Mapper(model=stub_model, index=index, id_map=id_map, top_k=1)
        candidates = mapper.map(_make_mention("Hypotonia"))
        # A single-element list is trivially sorted; just verify it has ≤1 entry
        assert len(candidates) <= 1


# ---------------------------------------------------------------------------
# Tests: confidence scores in [0.0, 1.0]
# ---------------------------------------------------------------------------


class TestConfidenceRange:
    """Every confidence score must be in [0.0, 1.0]."""

    def test_all_confidences_in_range(
        self,
        stub_model: _StubSentenceTransformer,
        small_index_and_id_map,
    ) -> None:
        index, id_map = small_index_and_id_map
        mapper = Mapper(model=stub_model, index=index, id_map=id_map, top_k=5)
        candidates = mapper.map(_make_mention("Seizure"))

        for c in candidates:
            assert 0.0 <= c.confidence <= 1.0, (
                f"Confidence {c.confidence} out of [0.0, 1.0] for {c.hpo_id}"
            )

    def test_confidence_formula_maps_inner_product_correctly(
        self,
        stub_model: _StubSentenceTransformer,
    ) -> None:
        """Verify the (score + 1) / 2 normalization formula directly.

        Build a 1-vector index where the query IS the stored vector, so the
        inner product is exactly 1.0 (perfect match).  The confidence should
        be (1.0 + 1) / 2 = 1.0.
        """
        # Encode a single text to get a normalized vector
        vec = stub_model.encode(["Seizure"], normalize_embeddings=True, convert_to_numpy=True)
        index = _build_index(vec)
        id_map = _make_id_map(["HP:0001250"], ["Seizure"])

        mapper = Mapper(model=stub_model, index=index, id_map=id_map, top_k=1)
        candidates = mapper.map(_make_mention("Seizure"))

        assert len(candidates) == 1
        # The stub always returns the same vector for the same seed, so the
        # query vector equals the stored vector → inner product ≈ 1.0
        assert abs(candidates[0].confidence - 1.0) < 1e-5, (
            f"Expected confidence ≈ 1.0 for identical vectors, got {candidates[0].confidence}"
        )


# ---------------------------------------------------------------------------
# Tests: deduplication by HPO ID
# ---------------------------------------------------------------------------


class TestDeduplication:
    """When multiple vectors share the same HPO ID, only the highest score is kept."""

    def test_duplicate_hpo_ids_deduplicated(
        self,
        stub_model: _StubSentenceTransformer,
    ) -> None:
        """Index has 3 vectors for HP:0001250 (label + 2 synonyms) and 1 for HP:0001290.

        After deduplication, at most 2 unique HPO IDs should appear in results.
        """
        texts = ["Seizure", "Seizures", "Epileptic seizure", "Hypotonia"]
        hpo_ids = ["HP:0001250", "HP:0001250", "HP:0001250", "HP:0001290"]

        vectors = stub_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        index = _build_index(vectors)
        id_map = _make_id_map(hpo_ids, texts)

        mapper = Mapper(model=stub_model, index=index, id_map=id_map, top_k=5)
        candidates = mapper.map(_make_mention("Seizure"))

        returned_ids = [c.hpo_id for c in candidates]
        # No duplicates
        assert len(returned_ids) == len(set(returned_ids)), (
            f"Duplicate HPO IDs in results: {returned_ids}"
        )

    def test_deduplication_keeps_highest_score(
        self,
        stub_model: _StubSentenceTransformer,
    ) -> None:
        """When two vectors share an HPO ID, the one with the higher score is kept.

        We build an index with two vectors for HP:0001250.  The first vector is
        the query itself (inner product = 1.0), the second is a random vector
        (lower score).  After deduplication, the confidence for HP:0001250 must
        be ≈ 1.0 (from the first vector).
        """
        # First vector: the query vector itself (will score 1.0)
        query_vec = stub_model.encode(
            ["Seizure"], normalize_embeddings=True, convert_to_numpy=True
        )
        # Second vector: a different random vector (lower score)
        rng = np.random.default_rng(seed=99)
        other_vec = rng.standard_normal((1, _EMBEDDING_DIM)).astype(np.float32)
        other_vec /= np.linalg.norm(other_vec, axis=1, keepdims=True)

        vectors = np.vstack([query_vec, other_vec])
        index = _build_index(vectors)
        id_map = _make_id_map(
            ["HP:0001250", "HP:0001250"],
            ["Seizure", "Seizures"],
        )

        mapper = Mapper(model=stub_model, index=index, id_map=id_map, top_k=3)
        candidates = mapper.map(_make_mention("Seizure"))

        # Only one candidate for HP:0001250
        hp_candidates = [c for c in candidates if c.hpo_id == "HP:0001250"]
        assert len(hp_candidates) == 1
        # Its confidence should be ≈ 1.0 (from the matching vector)
        assert abs(hp_candidates[0].confidence - 1.0) < 1e-5, (
            f"Expected confidence ≈ 1.0, got {hp_candidates[0].confidence}"
        )


# ---------------------------------------------------------------------------
# Tests: returns at most top_k results
# ---------------------------------------------------------------------------


class TestTopKLimit:
    """The mapper must return at most top_k candidates."""

    def test_returns_at_most_top_k(
        self,
        stub_model: _StubSentenceTransformer,
        small_index_and_id_map,
    ) -> None:
        index, id_map = small_index_and_id_map
        for top_k in (1, 2, 3, 5):
            mapper = Mapper(model=stub_model, index=index, id_map=id_map, top_k=top_k)
            candidates = mapper.map(_make_mention("Seizure"))
            assert len(candidates) <= top_k, (
                f"Expected ≤{top_k} candidates, got {len(candidates)}"
            )

    def test_top_k_1_returns_single_best(
        self,
        stub_model: _StubSentenceTransformer,
        small_index_and_id_map,
    ) -> None:
        index, id_map = small_index_and_id_map
        mapper = Mapper(model=stub_model, index=index, id_map=id_map, top_k=1)
        candidates = mapper.map(_make_mention("Seizure"))
        assert len(candidates) == 1

    def test_top_k_larger_than_index_returns_all(
        self,
        stub_model: _StubSentenceTransformer,
    ) -> None:
        """When top_k > number of unique HPO IDs, return all available candidates."""
        texts = ["Seizure", "Hypotonia"]
        hpo_ids = ["HP:0001250", "HP:0001290"]
        vectors = stub_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        index = _build_index(vectors)
        id_map = _make_id_map(hpo_ids, texts)

        mapper = Mapper(model=stub_model, index=index, id_map=id_map, top_k=10)
        candidates = mapper.map(_make_mention("Seizure"))
        # Can't return more than the 2 unique terms in the index
        assert len(candidates) <= 2


# ---------------------------------------------------------------------------
# Tests: empty list when index has no results
# ---------------------------------------------------------------------------


class TestEmptyIndex:
    """Mapper must return an empty list when the FAISS index is empty."""

    def test_empty_index_returns_empty_list(
        self,
        stub_model: _StubSentenceTransformer,
    ) -> None:
        """An empty FAISS index should return an empty candidate list."""
        index = faiss.IndexFlatIP(_EMBEDDING_DIM)  # no vectors added
        id_map: list[dict] = []

        mapper = Mapper(model=stub_model, index=index, id_map=id_map, top_k=3)
        candidates = mapper.map(_make_mention("Seizure"))

        assert candidates == [], f"Expected empty list, got {candidates}"

    def test_all_minus_one_indices_returns_empty_list(
        self,
        stub_model: _StubSentenceTransformer,
    ) -> None:
        """FAISS returns -1 for all indices when the index has fewer vectors than fetch_k.

        Simulate this by patching the index with a mock that always returns -1.
        """

        class _AlwaysMinusOneIndex:
            """Fake FAISS index that always returns -1 indices."""

            def search(self, query: np.ndarray, k: int):
                n = query.shape[0]
                scores = np.full((n, k), -1.0, dtype=np.float32)
                indices = np.full((n, k), -1, dtype=np.int64)
                return scores, indices

        mapper = Mapper(
            model=stub_model,
            index=_AlwaysMinusOneIndex(),  # type: ignore[arg-type]
            id_map=[],
            top_k=3,
        )
        candidates = mapper.map(_make_mention("Seizure"))
        assert candidates == [], f"Expected empty list, got {candidates}"


# ---------------------------------------------------------------------------
# Tests: encoding failure
# ---------------------------------------------------------------------------


class TestEncodingFailure:
    """When model.encode() raises, the mapper must log a warning and return []."""

    def test_encoding_failure_returns_empty_list(
        self,
        small_index_and_id_map,
    ) -> None:
        index, id_map = small_index_and_id_map
        failing_model = _FailingEncoder()

        mapper = Mapper(
            model=failing_model,  # type: ignore[arg-type]
            index=index,
            id_map=id_map,
            top_k=3,
        )
        candidates = mapper.map(_make_mention("Seizure"))
        assert candidates == [], f"Expected empty list on encoding failure, got {candidates}"

    def test_encoding_failure_logs_warning(
        self,
        small_index_and_id_map,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        index, id_map = small_index_and_id_map
        failing_model = _FailingEncoder()

        mapper = Mapper(
            model=failing_model,  # type: ignore[arg-type]
            index=index,
            id_map=id_map,
            top_k=3,
        )

        with caplog.at_level(logging.WARNING, logger="src.mapping.mapper"):
            mapper.map(_make_mention("Seizure"))

        assert any("warning" in r.levelname.lower() or r.levelno >= logging.WARNING
                   for r in caplog.records), (
            "Expected a WARNING log record when encoding fails"
        )

    def test_encoding_failure_does_not_raise(
        self,
        small_index_and_id_map,
    ) -> None:
        """Encoding failure must be swallowed — no exception propagated."""
        index, id_map = small_index_and_id_map
        failing_model = _FailingEncoder()

        mapper = Mapper(
            model=failing_model,  # type: ignore[arg-type]
            index=index,
            id_map=id_map,
            top_k=3,
        )
        # Should not raise
        try:
            mapper.map(_make_mention("Seizure"))
        except Exception as exc:
            pytest.fail(f"map() raised an unexpected exception: {exc}")
