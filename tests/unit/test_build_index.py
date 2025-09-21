"""
Unit tests for src/mapping/build_index.py.

Tests cover:
- build_faiss_index produces an index with the correct number of vectors
- The serialized ID map has the correct structure and length
- load_faiss_index raises IndexNotFoundError when files are missing
- Round-trip: build then load returns a consistent index and ID map

A stub SentenceTransformer that returns random normalized vectors of dim 768
is used to avoid loading any real model during unit tests.
"""

from __future__ import annotations

import json
import os
import re

import faiss
import numpy as np
import pytest

from src.data.build_hpo_db import HPODatabase, HPOTerm
from src.mapping.build_index import build_faiss_index, load_faiss_index
from src.pipeline import IndexNotFoundError

# ---------------------------------------------------------------------------
# Stub model
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 768


class _StubSentenceTransformer:
    """Minimal stub that mimics SentenceTransformer.encode().

    Returns random float32 vectors of dimension 768, L2-normalized so that
    the inner product between any two vectors is in [-1, 1].  The
    ``normalize_embeddings`` kwarg is accepted but ignored (vectors are
    always normalized).
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


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_model() -> _StubSentenceTransformer:
    """Return a fresh stub model instance."""
    return _StubSentenceTransformer()


@pytest.fixture()
def minimal_hpo_db() -> HPODatabase:
    """A minimal HPODatabase with 3 terms and varying synonym counts.

    Term breakdown:
        HP:0001250  Seizure          + 2 synonyms  → 3 vectors
        HP:0001290  Hypotonia        + 1 synonym   → 2 vectors
        HP:0000365  Hearing loss     + 0 synonyms  → 1 vector
    Total expected vectors: 6
    """
    return HPODatabase(
        version="test",
        terms=[
            HPOTerm(
                id="HP:0001250",
                label="Seizure",
                synonyms=["Seizures", "Epileptic seizure"],
                definition="An episode of abnormal electrical activity in the brain.",
            ),
            HPOTerm(
                id="HP:0001290",
                label="Hypotonia",
                synonyms=["Decreased muscle tone"],
                definition="Reduced muscle tone.",
            ),
            HPOTerm(
                id="HP:0000365",
                label="Hearing loss",
                synonyms=[],
                definition="A reduction in the ability to perceive sounds.",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Helper: build index into a temp directory and return paths
# ---------------------------------------------------------------------------


def _build_to_tempdir(
    hpo_db: HPODatabase,
    model: _StubSentenceTransformer,
    tmp_path: str,
) -> tuple[str, str]:
    """Build the FAISS index and ID map into *tmp_path* and return (index_path, id_map_path)."""
    index_path = os.path.join(tmp_path, "test.faiss")
    id_map_path = os.path.join(tmp_path, "test_id_map.json")
    build_faiss_index(hpo_db, model, index_path, id_map_path)
    return index_path, id_map_path


# ---------------------------------------------------------------------------
# Tests: build_faiss_index
# ---------------------------------------------------------------------------


class TestBuildFaissIndex:
    """Tests for the build_faiss_index function."""

    def test_index_vector_count_matches_labels_and_synonyms(
        self,
        minimal_hpo_db: HPODatabase,
        stub_model: _StubSentenceTransformer,
        tmp_path,
    ) -> None:
        """The FAISS index must contain exactly one vector per label + synonym."""
        expected_count = len(minimal_hpo_db.all_labels_and_synonyms())  # 6
        index_path, _ = _build_to_tempdir(minimal_hpo_db, stub_model, str(tmp_path))

        index = faiss.read_index(index_path)
        assert index.ntotal == expected_count, (
            f"Expected {expected_count} vectors in index, got {index.ntotal}"
        )

    def test_id_map_length_matches_labels_and_synonyms(
        self,
        minimal_hpo_db: HPODatabase,
        stub_model: _StubSentenceTransformer,
        tmp_path,
    ) -> None:
        """The ID map must have the same number of entries as labels + synonyms."""
        expected_count = len(minimal_hpo_db.all_labels_and_synonyms())  # 6
        _, id_map_path = _build_to_tempdir(minimal_hpo_db, stub_model, str(tmp_path))

        with open(id_map_path, "r", encoding="utf-8") as fh:
            id_map = json.load(fh)

        assert len(id_map) == expected_count, (
            f"Expected {expected_count} ID map entries, got {len(id_map)}"
        )

    def test_id_map_entry_structure(
        self,
        minimal_hpo_db: HPODatabase,
        stub_model: _StubSentenceTransformer,
        tmp_path,
    ) -> None:
        """Every ID map entry must have 'vector_idx', 'hpo_id', and 'text' keys."""
        _, id_map_path = _build_to_tempdir(minimal_hpo_db, stub_model, str(tmp_path))

        with open(id_map_path, "r", encoding="utf-8") as fh:
            id_map = json.load(fh)

        required_keys = {"vector_idx", "hpo_id", "text"}
        for i, entry in enumerate(id_map):
            missing = required_keys - entry.keys()
            assert not missing, (
                f"ID map entry {i} is missing keys: {missing}"
            )

    def test_id_map_vector_idx_is_sequential(
        self,
        minimal_hpo_db: HPODatabase,
        stub_model: _StubSentenceTransformer,
        tmp_path,
    ) -> None:
        """vector_idx values must be 0, 1, 2, … in order."""
        _, id_map_path = _build_to_tempdir(minimal_hpo_db, stub_model, str(tmp_path))

        with open(id_map_path, "r", encoding="utf-8") as fh:
            id_map = json.load(fh)

        for expected_idx, entry in enumerate(id_map):
            assert entry["vector_idx"] == expected_idx, (
                f"Expected vector_idx={expected_idx}, got {entry['vector_idx']}"
            )

    def test_id_map_hpo_ids_are_valid(
        self,
        minimal_hpo_db: HPODatabase,
        stub_model: _StubSentenceTransformer,
        tmp_path,
    ) -> None:
        """Every hpo_id in the ID map must match the HP:\\d+ pattern."""
        hpo_id_pattern = re.compile(r"^HP:\d+$")
        _, id_map_path = _build_to_tempdir(minimal_hpo_db, stub_model, str(tmp_path))

        with open(id_map_path, "r", encoding="utf-8") as fh:
            id_map = json.load(fh)

        for entry in id_map:
            assert hpo_id_pattern.match(entry["hpo_id"]), (
                f"Invalid hpo_id format: {entry['hpo_id']!r}"
            )

    def test_id_map_texts_match_labels_and_synonyms(
        self,
        minimal_hpo_db: HPODatabase,
        stub_model: _StubSentenceTransformer,
        tmp_path,
    ) -> None:
        """The 'text' values in the ID map must match the labels/synonyms from the DB."""
        expected_pairs = minimal_hpo_db.all_labels_and_synonyms()
        expected_texts = [text for text, _ in expected_pairs]

        _, id_map_path = _build_to_tempdir(minimal_hpo_db, stub_model, str(tmp_path))

        with open(id_map_path, "r", encoding="utf-8") as fh:
            id_map = json.load(fh)

        actual_texts = [entry["text"] for entry in id_map]
        assert actual_texts == expected_texts

    def test_index_files_are_created(
        self,
        minimal_hpo_db: HPODatabase,
        stub_model: _StubSentenceTransformer,
        tmp_path,
    ) -> None:
        """Both output files must exist after build_faiss_index completes."""
        index_path, id_map_path = _build_to_tempdir(
            minimal_hpo_db, stub_model, str(tmp_path)
        )
        assert os.path.exists(index_path), f"Index file not found: {index_path}"
        assert os.path.exists(id_map_path), f"ID map file not found: {id_map_path}"

    def test_single_term_no_synonyms(
        self,
        stub_model: _StubSentenceTransformer,
        tmp_path,
    ) -> None:
        """A database with a single term and no synonyms should produce exactly 1 vector."""
        db = HPODatabase(
            version="test",
            terms=[
                HPOTerm(
                    id="HP:0000001",
                    label="All",
                    synonyms=[],
                    definition="Root term.",
                )
            ],
        )
        index_path, id_map_path = _build_to_tempdir(db, stub_model, str(tmp_path))

        index = faiss.read_index(index_path)
        assert index.ntotal == 1

        with open(id_map_path, "r", encoding="utf-8") as fh:
            id_map = json.load(fh)
        assert len(id_map) == 1

    def test_index_dimension_matches_embedding_dim(
        self,
        minimal_hpo_db: HPODatabase,
        stub_model: _StubSentenceTransformer,
        tmp_path,
    ) -> None:
        """The FAISS index dimension must match the stub model's output dimension (768)."""
        index_path, _ = _build_to_tempdir(minimal_hpo_db, stub_model, str(tmp_path))
        index = faiss.read_index(index_path)
        assert index.d == _EMBEDDING_DIM, (
            f"Expected index dimension {_EMBEDDING_DIM}, got {index.d}"
        )


# ---------------------------------------------------------------------------
# Tests: load_faiss_index
# ---------------------------------------------------------------------------


class TestLoadFaissIndex:
    """Tests for the load_faiss_index function."""

    def test_load_raises_when_index_file_missing(self, tmp_path) -> None:
        """load_faiss_index must raise IndexNotFoundError when the index file is absent."""
        missing_index = str(tmp_path / "nonexistent.faiss")
        id_map_path = str(tmp_path / "id_map.json")
        with open(id_map_path, "w") as fh:
            json.dump([], fh)

        with pytest.raises(IndexNotFoundError, match="nonexistent.faiss"):
            load_faiss_index(missing_index, id_map_path)

    def test_load_raises_when_id_map_file_missing(
        self,
        minimal_hpo_db: HPODatabase,
        stub_model: _StubSentenceTransformer,
        tmp_path,
    ) -> None:
        """load_faiss_index must raise IndexNotFoundError when the ID map file is absent."""
        index_path, _ = _build_to_tempdir(minimal_hpo_db, stub_model, str(tmp_path))
        missing_id_map = str(tmp_path / "nonexistent_id_map.json")

        with pytest.raises(IndexNotFoundError, match="nonexistent_id_map.json"):
            load_faiss_index(index_path, missing_id_map)

    def test_load_raises_when_both_files_missing(self, tmp_path) -> None:
        """load_faiss_index must raise IndexNotFoundError when both files are absent."""
        with pytest.raises(IndexNotFoundError):
            load_faiss_index(
                str(tmp_path / "missing.faiss"),
                str(tmp_path / "missing_id_map.json"),
            )

    def test_load_returns_correct_types(
        self,
        minimal_hpo_db: HPODatabase,
        stub_model: _StubSentenceTransformer,
        tmp_path,
    ) -> None:
        """load_faiss_index must return a (faiss.Index, list) tuple."""
        index_path, id_map_path = _build_to_tempdir(
            minimal_hpo_db, stub_model, str(tmp_path)
        )
        index, id_map = load_faiss_index(index_path, id_map_path)

        assert isinstance(index, faiss.Index)
        assert isinstance(id_map, list)

    def test_load_index_ntotal_matches_build(
        self,
        minimal_hpo_db: HPODatabase,
        stub_model: _StubSentenceTransformer,
        tmp_path,
    ) -> None:
        """The loaded index must have the same ntotal as the built index."""
        expected_count = len(minimal_hpo_db.all_labels_and_synonyms())
        index_path, id_map_path = _build_to_tempdir(
            minimal_hpo_db, stub_model, str(tmp_path)
        )
        index, id_map = load_faiss_index(index_path, id_map_path)

        assert index.ntotal == expected_count
        assert len(id_map) == expected_count

    def test_load_id_map_structure_preserved(
        self,
        minimal_hpo_db: HPODatabase,
        stub_model: _StubSentenceTransformer,
        tmp_path,
    ) -> None:
        """The loaded ID map must have the same structure as the built one."""
        index_path, id_map_path = _build_to_tempdir(
            minimal_hpo_db, stub_model, str(tmp_path)
        )
        _, id_map = load_faiss_index(index_path, id_map_path)

        required_keys = {"vector_idx", "hpo_id", "text"}
        for entry in id_map:
            assert required_keys.issubset(entry.keys())
