"""
FAISS index builder for the Clinical Phenotype Extraction and HPO Mapping Pipeline.

This module provides:
- build_faiss_index(): encode all HPO labels and synonyms with a SentenceTransformer
  model, L2-normalize vectors, build a FAISS IndexFlatIP, and serialize both the
  index and the ID map to disk.
- load_faiss_index(): load a serialized FAISS index and ID map from disk.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.data.build_hpo_db import HPODatabase
from src.pipeline import IndexNotFoundError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_faiss_index(
    hpo_db: HPODatabase,
    model: SentenceTransformer,
    output_index_path: str,
    output_id_map_path: str,
) -> None:
    """Encode all HPO labels and synonyms with the given model, L2-normalize vectors,
    build a FAISS IndexFlatIP, serialize index and ID map to disk.

    The ID map is a JSON list where each entry has:
        - ``vector_idx``: integer position in the FAISS index
        - ``hpo_id``: HPO term identifier (e.g. "HP:0001250")
        - ``text``: the label or synonym text that was encoded

    Args:
        hpo_db: Loaded HPODatabase containing all terms.
        model: A SentenceTransformer model used to encode text.
        output_index_path: File path where the FAISS index will be written.
        output_id_map_path: File path where the JSON ID map will be written.
    """
    # Collect all (text, hpo_id) pairs from labels and synonyms
    pairs: list[tuple[str, str]] = hpo_db.all_labels_and_synonyms()

    texts: list[str] = [text for text, _ in pairs]
    hpo_ids: list[str] = [hpo_id for _, hpo_id in pairs]

    logger.info("Encoding %d texts (labels + synonyms) …", len(texts))

    # Encode all texts in a single batch with L2 normalization applied by the model.
    # normalize_embeddings=True handles L2 normalization automatically so that
    # inner product == cosine similarity in the FAISS IndexFlatIP.
    vectors: np.ndarray = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Ensure float32 — FAISS requires it
    vectors = vectors.astype(np.float32)

    # Build the index
    dimension: int = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    logger.info(
        "FAISS index built: total_vectors_encoded=%d, index_ntotal=%d, dimension=%d",
        len(texts),
        index.ntotal,
        dimension,
    )

    # Serialize the FAISS index to disk
    faiss.write_index(index, output_index_path)
    logger.info("FAISS index written to %s", output_index_path)

    # Build and serialize the ID map
    id_map: list[dict] = [
        {"vector_idx": i, "hpo_id": hpo_ids[i], "text": texts[i]}
        for i in range(len(texts))
    ]

    with open(output_id_map_path, "w", encoding="utf-8") as fh:
        json.dump(id_map, fh, ensure_ascii=False, indent=2)

    logger.info("ID map written to %s (%d entries)", output_id_map_path, len(id_map))


def load_faiss_index(
    index_path: str,
    id_map_path: str,
) -> tuple[faiss.Index, list[dict]]:
    """Load a serialized FAISS index and ID map from disk.

    Args:
        index_path: Path to the serialized FAISS index file.
        id_map_path: Path to the JSON ID map file.

    Returns:
        A tuple of (faiss.Index, list[dict]) where each dict in the list has
        keys ``vector_idx``, ``hpo_id``, and ``text``.

    Raises:
        IndexNotFoundError: If either file is missing.
    """
    import os

    if not os.path.exists(index_path):
        raise IndexNotFoundError(
            f"FAISS index file not found: {index_path}"
        )
    if not os.path.exists(id_map_path):
        raise IndexNotFoundError(
            f"FAISS ID map file not found: {id_map_path}"
        )

    index = faiss.read_index(index_path)
    logger.info(
        "FAISS index loaded from %s (ntotal=%d)", index_path, index.ntotal
    )

    with open(id_map_path, "r", encoding="utf-8") as fh:
        id_map: list[dict] = json.load(fh)

    logger.info("ID map loaded from %s (%d entries)", id_map_path, len(id_map))

    return index, id_map
