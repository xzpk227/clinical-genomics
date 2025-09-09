"""
HPO Database builder for the Clinical Phenotype Extraction and HPO Mapping Pipeline.

This module provides:
- HPOTerm and HPODatabase dataclasses
- build_hpo_database(): parse hp.json, filter to phenotypic abnormality subtree,
  and serialize to a JSON file
- load_hpo_database(): load a serialized HPODatabase from a JSON file
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.pipeline import HPODatabaseError

logger = logging.getLogger(__name__)

# Root of the phenotypic abnormality subtree
_PHENOTYPIC_ABNORMALITY_ROOT = "HP:0000118"

# Regex to extract the date portion from a version URL like:
# "http://purl.obolibrary.org/obo/hp/releases/2024-04-26/hp.json"
_VERSION_DATE_RE = re.compile(r"/releases/(\d{4}-\d{2}-\d{2})/")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HPOTerm:
    """A single HPO term with its ID, label, synonyms, and definition."""

    id: str           # e.g. "HP:0001250"
    label: str        # e.g. "Seizure"
    synonyms: list[str] = field(default_factory=list)
    definition: str = ""


@dataclass
class HPODatabase:
    """A structured collection of HPO terms."""

    version: str
    terms: list[HPOTerm] = field(default_factory=list)

    def get_by_id(self, hpo_id: str) -> Optional[HPOTerm]:
        """Return the HPOTerm with the given ID, or None if not found."""
        for term in self.terms:
            if term.id == hpo_id:
                return term
        return None

    def all_labels_and_synonyms(self) -> list[tuple[str, str]]:
        """Return [(text, hpo_id), ...] for all labels and synonyms across all terms."""
        result: list[tuple[str, str]] = []
        for term in self.terms:
            result.append((term.label, term.id))
            for synonym in term.synonyms:
                result.append((synonym, term.id))
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _node_id_to_hpo(node_id: str) -> str:
    """Convert a node URI to an HPO ID string.

    Example:
        "http://purl.obolibrary.org/obo/HP_0001250" -> "HP:0001250"
    """
    # Extract the local part after the last '/'
    local = node_id.rsplit("/", 1)[-1]
    # Replace underscore separator with colon
    return local.replace("HP_", "HP:", 1)


def _collect_subtree(root_id: str, edges: list[dict]) -> set[str]:
    """Return the set of all HPO IDs that are descendants of root_id (inclusive).

    Traverses 'is_a' edges in the graph to find all descendants.
    """
    # Build a parent -> children mapping from is_a edges
    children: dict[str, list[str]] = {}
    for edge in edges:
        if edge.get("pred") != "is_a":
            continue
        sub = _node_id_to_hpo(edge["sub"])
        obj = _node_id_to_hpo(edge["obj"])
        children.setdefault(obj, []).append(sub)

    # BFS/DFS from root
    subtree: set[str] = set()
    stack = [root_id]
    while stack:
        current = stack.pop()
        if current in subtree:
            continue
        subtree.add(current)
        for child in children.get(current, []):
            stack.append(child)

    return subtree


def _extract_version(version_url: str) -> str:
    """Extract the date portion from a version URL.

    Example:
        "http://purl.obolibrary.org/obo/hp/releases/2024-04-26/hp.json"
        -> "2024-04-26"

    Falls back to the full URL string if the pattern is not found.
    """
    match = _VERSION_DATE_RE.search(version_url)
    if match:
        return match.group(1)
    return version_url


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------


def build_hpo_database(source_path: str, output_path: str) -> HPODatabase:
    """Parse hp.json, filter to phenotypic abnormality subtree (HP:0000118),
    extract id, name, synonyms, and def for each term, log HPO release version
    and term count, and serialize to output_path as JSON.

    Args:
        source_path: Path to the hp.json source file from HPO GitHub releases.
        output_path: Path where the serialized hpo_database.json will be written.

    Returns:
        The constructed HPODatabase.

    Raises:
        HPODatabaseError: If the source file is missing or malformed.
    """
    # Load source file
    try:
        with open(source_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        raise HPODatabaseError(
            f"HPO source file not found: {source_path}"
        )
    except json.JSONDecodeError as exc:
        raise HPODatabaseError(
            f"HPO source file is not valid JSON ({source_path}): {exc}"
        )

    # Validate top-level structure
    try:
        graphs = raw["graphs"]
        if not graphs:
            raise HPODatabaseError(
                f"HPO source file has no graphs: {source_path}"
            )
        graph = graphs[0]
    except (KeyError, TypeError) as exc:
        raise HPODatabaseError(
            f"HPO source file has unexpected structure ({source_path}): {exc}"
        )

    # Extract version
    try:
        version_url: str = graph["meta"]["version"]
        version = _extract_version(version_url)
    except (KeyError, TypeError):
        version = "unknown"

    # Collect edges and determine the phenotypic abnormality subtree
    edges: list[dict] = graph.get("edges", [])
    subtree_ids = _collect_subtree(_PHENOTYPIC_ABNORMALITY_ROOT, edges)

    # Parse nodes
    nodes: list[dict] = graph.get("nodes", [])
    terms: list[HPOTerm] = []

    for node in nodes:
        # Only process CLASS nodes
        if node.get("type") != "CLASS":
            continue

        node_id_raw: str = node.get("id", "")
        hpo_id = _node_id_to_hpo(node_id_raw)

        # Must be in the phenotypic abnormality subtree
        if hpo_id not in subtree_ids:
            continue

        # Must have a label
        label: str = node.get("lbl", "").strip()
        if not label:
            continue

        # Extract synonyms (all synonym types)
        meta: dict = node.get("meta", {}) or {}
        raw_synonyms: list[dict] = meta.get("synonyms", []) or []
        synonyms: list[str] = [
            s["val"]
            for s in raw_synonyms
            if isinstance(s, dict) and s.get("val")
        ]

        # Extract definition
        definition_obj = meta.get("definition")
        if isinstance(definition_obj, dict):
            definition: str = definition_obj.get("val", "") or ""
        else:
            definition = ""

        terms.append(
            HPOTerm(
                id=hpo_id,
                label=label,
                synonyms=synonyms,
                definition=definition,
            )
        )

    db = HPODatabase(version=version, terms=terms)

    # Log version and term count (Requirement 4.3)
    logger.info(
        "HPO database built: version=%s, term_count=%d",
        db.version,
        len(db.terms),
    )

    # Serialize to output_path
    serialized = {
        "version": db.version,
        "term_count": len(db.terms),
        "terms": [
            {
                "id": t.id,
                "label": t.label,
                "synonyms": t.synonyms,
                "definition": t.definition,
            }
            for t in db.terms
        ],
    }

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(serialized, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise HPODatabaseError(
            f"Failed to write HPO database to {output_path}: {exc}"
        )

    return db


# ---------------------------------------------------------------------------
# Load function
# ---------------------------------------------------------------------------


def load_hpo_database(path: str) -> HPODatabase:
    """Load a serialized HPODatabase from a JSON file.

    Args:
        path: Path to the serialized hpo_database.json file.

    Returns:
        The loaded HPODatabase.

    Raises:
        HPODatabaseError: If the file is missing or malformed.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise HPODatabaseError(
            f"HPO database file not found: {path}"
        )
    except json.JSONDecodeError as exc:
        raise HPODatabaseError(
            f"HPO database file is not valid JSON ({path}): {exc}"
        )

    # Validate required top-level keys
    if not isinstance(data, dict):
        raise HPODatabaseError(
            f"HPO database file has unexpected format (expected a JSON object): {path}"
        )

    missing = [k for k in ("version", "terms") if k not in data]
    if missing:
        raise HPODatabaseError(
            f"HPO database file is missing required keys {missing}: {path}"
        )

    version: str = data["version"]
    raw_terms = data["terms"]

    if not isinstance(raw_terms, list):
        raise HPODatabaseError(
            f"HPO database 'terms' field must be a list: {path}"
        )

    terms: list[HPOTerm] = []
    for i, raw in enumerate(raw_terms):
        if not isinstance(raw, dict):
            raise HPODatabaseError(
                f"HPO database term at index {i} is not a JSON object: {path}"
            )
        try:
            terms.append(
                HPOTerm(
                    id=raw["id"],
                    label=raw["label"],
                    synonyms=raw.get("synonyms", []),
                    definition=raw.get("definition", ""),
                )
            )
        except KeyError as exc:
            raise HPODatabaseError(
                f"HPO database term at index {i} is missing required key {exc}: {path}"
            )

    return HPODatabase(version=version, terms=terms)
