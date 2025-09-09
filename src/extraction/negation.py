"""
Negation handler for the Clinical Phenotype Extraction and HPO Mapping Pipeline.

Uses negspaCy's NegEx implementation to detect negated phenotype mentions in
clinical notes. Supports a configurable negation cue list loaded from
``config/negation_cues.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import spacy
import spacy.tokens
import spacy.util
from negspacy.negation import Negex  # noqa: F401 — registers the 'negex' spaCy factory

from src.extraction.extractor import Mention

logger = logging.getLogger(__name__)

# Default path to the negation cue configuration file.
_DEFAULT_CUES_PATH = Path(__file__).parent.parent.parent / "config" / "negation_cues.json"


class NegationHandler:
    """Detects negated phenotype mentions using the NegEx algorithm via negspaCy.

    At construction time, the handler builds its own spaCy pipeline with the
    ``negex`` component added so that ``span._.negex`` is available on every
    span created from the processed document.

    The negation cue list is loaded from ``config/negation_cues.json`` by
    default, but a custom list can be supplied directly via ``cue_list``.
    """

    def __init__(self, cue_list: list[str] | None = None) -> None:
        """Load negation cues and initialise negspaCy's Negex component.

        When ``cue_list`` is ``None``, the cues are loaded from
        ``config/negation_cues.json``.  The JSON file must have
        ``"preceding_negations"`` and ``"following_negations"`` keys.

        The combined cue list is passed to negspaCy's ``Negex`` via a custom
        ``neg_termset`` so that the NegEx window scan uses the project-specific
        vocabulary.

        Args:
            cue_list: Optional flat list of negation cue strings.  When
                provided, these are split evenly between preceding and
                following negations (all treated as preceding for simplicity).
                When ``None``, cues are loaded from the JSON config file.
        """
        if cue_list is not None:
            preceding = list(cue_list)
            following: list[str] = []
        else:
            preceding, following = self._load_cues_from_file(_DEFAULT_CUES_PATH)

        self._preceding = preceding
        self._following = following

        # Build a spaCy pipeline with sentencizer + negex components.
        # negspaCy's NegEx requires sentence boundaries to be set.
        self._nlp = spacy.blank("en")
        self._nlp.add_pipe("sentencizer")

        # Pass the cue lists directly as the neg_termset config dict.
        # This is the most reliable approach and avoids any termset API
        # version differences.
        neg_termset = {
            "pseudo_negations": [],
            "preceding_negations": preceding,
            "following_negations": following,
            "termination": ["but", "however", "though", "except"],
        }

        self._nlp.add_pipe(
            "negex",
            config={"neg_termset": neg_termset},
        )

        # Store a direct reference to the negex component so we can call it
        # on a doc that already has entities set (rather than running the full
        # pipeline which would overwrite doc.ents).
        self._negex_pipe = self._nlp.get_pipe("negex")
        self._sentencizer_pipe = self._nlp.get_pipe("sentencizer")

        logger.debug(
            "NegationHandler initialised with %d preceding and %d following cues.",
            len(preceding),
            len(following),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(
        self,
        doc: spacy.tokens.Doc,
        mentions: list[Mention],
    ) -> list[Mention]:
        """Check each mention's span for negation and set ``mention.negated``.

        The method re-processes the document text through the internal spaCy
        pipeline (which includes the ``negex`` component).  negspaCy's Negex
        component operates on ``doc.ents``, so each mention span is set as an
        entity in the document before the pipeline runs.  After processing,
        ``e._.negex`` is checked for each entity and the corresponding
        ``Mention.negated`` flag is set.

        All mentions are returned — negated and non-negated alike.  Mentions
        are never dropped.

        Args:
            doc: A spaCy ``Doc`` produced from the clinical note text.  The
                 doc's text is used to re-process through the negex pipeline.
            mentions: List of ``Mention`` objects to annotate.

        Returns:
            The same list of mentions with ``negated`` set appropriately.
        """
        if not mentions:
            return mentions

        # Process the text through the full pipeline (sentencizer sets sentence
        # boundaries, which negspaCy requires). We disable negex on the first
        # pass so we can set entities manually, then run negex separately.
        with self._nlp.select_pipes(disable=["negex"]):
            base_doc = self._nlp(doc.text)

        # Resolve each mention to a Span and build the entity list.
        spans: list[spacy.tokens.Span | None] = []
        for mention in mentions:
            span = base_doc.char_span(mention.start, mention.end, label="MENTION")
            if span is None:
                span = self._fallback_span(base_doc, mention.start, mention.end)
            spans.append(span)

        # Set non-None spans as doc.ents (negspaCy reads from doc.ents).
        valid_spans = [s for s in spans if s is not None]
        if valid_spans:
            base_doc.set_ents(spacy.util.filter_spans(valid_spans))

        # Run the negex component over the doc (it annotates doc.ents in-place).
        negex_doc = self._negex_pipe(base_doc)

        # Build a lookup from (start_char, end_char) → negex result.
        negex_map: dict[tuple[int, int], bool] = {
            (ent.start_char, ent.end_char): bool(ent._.negex)
            for ent in negex_doc.ents
        }

        # Propagate negation flags back to the original Mention objects.
        for mention, span in zip(mentions, spans):
            if span is not None:
                key = (span.start_char, span.end_char)
                if negex_map.get(key, False):
                    mention.negated = True

        return mentions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_cues_from_file(path: Path) -> tuple[list[str], list[str]]:
        """Load negation cues from a JSON file.

        Args:
            path: Path to the JSON file with ``"preceding_negations"`` and
                  ``"following_negations"`` keys.

        Returns:
            A tuple ``(preceding_negations, following_negations)``.

        Raises:
            FileNotFoundError: If the cue file does not exist.
            ValueError: If the JSON is malformed or missing required keys.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Negation cue file not found: {path}. "
                "Ensure config/negation_cues.json exists."
            )

        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Negation cue file is not valid JSON: {path}"
            ) from exc

        preceding = data.get("preceding_negations", [])
        following = data.get("following_negations", [])

        if not isinstance(preceding, list) or not isinstance(following, list):
            raise ValueError(
                "negation_cues.json must have list values for "
                "'preceding_negations' and 'following_negations'."
            )

        return preceding, following

    @staticmethod
    def _fallback_span(
        doc: spacy.tokens.Doc,
        start_char: int,
        end_char: int,
    ) -> spacy.tokens.Span | None:
        """Return a span covering tokens that overlap with the character range.

        Used when ``doc.char_span`` returns ``None`` due to alignment issues
        (e.g., the mention boundary falls inside a token).

        Args:
            doc: The spaCy document.
            start_char: Character offset of the start of the mention.
            end_char: Character offset of the end of the mention.

        Returns:
            A ``Span`` covering the overlapping tokens, or ``None`` if no
            tokens overlap with the range.
        """
        start_token: int | None = None
        end_token: int | None = None

        for token in doc:
            if token.idx < end_char and (token.idx + len(token.text)) > start_char:
                if start_token is None:
                    start_token = token.i
                end_token = token.i + 1

        if start_token is None or end_token is None:
            return None

        return spacy.tokens.Span(doc, start_token, end_token, label="MENTION")
