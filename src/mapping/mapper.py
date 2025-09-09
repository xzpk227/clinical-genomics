"""
Embedding-based HPO term mapper for the Clinical Phenotype Extraction and HPO Mapping Pipeline.

This module provides:
- HPOCandidate dataclass representing a single HPO mapping candidate
- Mapper class that encodes a Mention with BioLORD-2023, queries a FAISS index,
  deduplicates by HPO ID, normalizes scores, and returns ranked HPOCandidate lists
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.extraction.extractor import Mention

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class HPOCandidate:
    """A single HPO mapping candidate returned by the Mapper.

    Attributes:
        hpo_id:     HPO term identifier, e.g. "HP:0001250".
        hpo_label:  The label or synonym text that was encoded for this entry.
        confidence: Normalized similarity score in [0.0, 1.0].
    """

    hpo_id: str
    hpo_label: str
    confidence: float  # [0.0, 1.0]


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------


class Mapper:
    """Embedding-based HPO term mapper using BioLORD-2023 and FAISS.

    Encodes a Mention's text into a vector, queries the FAISS index for the
    nearest neighbours, deduplicates by HPO ID (keeping the highest score per
    term), normalizes inner-product scores to [0.0, 1.0], and returns a ranked
    list of HPOCandidate objects.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        index: faiss.Index,
        id_map: list[dict],
        top_k: int = 3,
    ) -> None:
        """Store all components needed for mapping.

        Args:
            model:   A SentenceTransformer model (e.g. BioLORD-2023) used to
                     encode mention text.
            index:   A FAISS index (IndexFlatIP) built from L2-normalized vectors.
            id_map:  List of dicts with keys ``vector_idx``, ``hpo_id``, and
                     ``text``, mapping each FAISS position to an HPO term.
            top_k:   Maximum number of candidates to return per mention.
        """
        self._model = model
        self._index = index
        self._id_map = id_map
        self._top_k = top_k

    def map(self, mention: Mention) -> list[HPOCandidate]:
        """Map a phenotype mention to a ranked list of HPO candidates.

        Steps:
        1. Encode ``mention.text`` with L2 normalization.
        2. Query FAISS for ``top_k * 3`` results to allow deduplication.
        3. Deduplicate by HPO ID, keeping the highest score per unique term.
        4. Normalize scores: ``confidence = (score + 1) / 2`` (inner product of
           L2-normalized vectors is in [-1, 1], this maps to [0.0, 1.0]).
        5. Clip confidence to [0.0, 1.0] to handle floating-point edge cases.
        6. Sort by confidence descending and return at most ``top_k`` candidates.

        Returns an empty list (no error) when:
        - The FAISS index is empty or all returned indices are -1.
        - The mention text cannot be encoded (logs a warning and skips).

        Args:
            mention: The phenotype mention to map.

        Returns:
            A list of at most ``top_k`` HPOCandidate objects sorted by
            confidence descending.
        """
        # Step 1: Encode the mention text
        try:
            query_vector: np.ndarray = self._model.encode(
                [mention.text],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except Exception:
            logger.warning(
                "Failed to encode mention text %r — skipping.",
                mention.text,
            )
            return []

        # Ensure float32 for FAISS
        query_vector = query_vector.astype(np.float32)

        # Step 2: Query FAISS — fetch extra results to allow deduplication
        fetch_k = self._top_k * 3
        scores, indices = self._index.search(query_vector, fetch_k)

        # scores and indices have shape (1, fetch_k)
        scores_flat = scores[0]
        indices_flat = indices[0]

        # Step 3: Deduplicate by HPO ID — keep highest score per unique hpo_id
        best_per_hpo: dict[str, tuple[float, str]] = {}  # hpo_id -> (score, text)

        for score, idx in zip(scores_flat, indices_flat):
            # FAISS returns -1 for empty slots (when index has fewer vectors than fetch_k)
            if idx == -1:
                continue

            entry = self._id_map[idx]
            hpo_id: str = entry["hpo_id"]
            text: str = entry["text"]

            if hpo_id not in best_per_hpo or score > best_per_hpo[hpo_id][0]:
                best_per_hpo[hpo_id] = (float(score), text)

        if not best_per_hpo:
            return []

        # Steps 4 & 5: Normalize and clip scores
        candidates: list[HPOCandidate] = []
        for hpo_id, (score, hpo_label) in best_per_hpo.items():
            confidence = (score + 1.0) / 2.0
            confidence = float(np.clip(confidence, 0.0, 1.0))
            candidates.append(
                HPOCandidate(
                    hpo_id=hpo_id,
                    hpo_label=hpo_label,
                    confidence=confidence,
                )
            )

        # Step 6: Sort by confidence descending, return at most top_k
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates[: self._top_k]
