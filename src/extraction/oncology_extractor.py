"""
Oncology-specific symptom and adverse event extractor.

Extends beyond HPO to detect oncology-relevant symptoms, toxicities, and
adverse events using a curated vocabulary of clinical synonyms.  Integrates
with the existing NegationHandler to mark negated mentions.

Symptom categories
------------------
  fatigue        – fatigue, asthenia, tiredness, weakness, lethargy, exhaustion
  nausea         – nausea, queasiness, nauseous, sick to stomach
  vomiting       – vomiting, emesis, retching
  fever          – fever, pyrexia, febrile, elevated temperature
  neuropathy     – peripheral neuropathy, paresthesia, numbness, tingling,
                   pins and needles, burning sensation in hands/feet
  dyspnea        – dyspnea, shortness of breath, breathlessness,
                   difficulty breathing, respiratory distress
  pain           – pain (with negation awareness)
  bleeding       – bleeding, hemorrhage, hematuria, hemoptysis, epistaxis,
                   bruising, ecchymosis, petechiae
  infection      – infection, sepsis, cellulitis, pneumonia, bacteremia
  thrombosis     – thrombosis, DVT, pulmonary embolism, clot
  pneumonitis    – pneumonitis, checkpoint pneumonitis, interstitial lung disease
  colitis        – colitis, immune-related colitis, checkpoint colitis
  hepatotoxicity – hepatotoxicity, elevated liver enzymes, transaminase elevation
  mucositis      – mucositis, stomatitis, mouth sores, mouth ulcers
  alopecia       – alopecia, hair loss
  rash           – rash, dermatitis, pruritus, urticaria, skin eruption

CTCAE grade language mapping (if grade word present in ±20-token window)
  mild / grade 1  → grade 1
  moderate / grade 2  → grade 2
  severe / grade 3  → grade 3
  life-threatening / grade 4  → grade 4
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import spacy
import spacy.util
from spacy.matcher import PhraseMatcher

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

ONCOLOGY_VOCABULARY: Dict[str, List[str]] = {
    "fatigue": [
        "fatigue", "asthenia", "tiredness", "lethargy", "exhaustion",
        "weakness", "generalized weakness", "cancer-related fatigue",
        "treatment-related fatigue",
    ],
    "nausea": [
        "nausea", "nauseous", "queasiness", "queasy",
        "chemotherapy-induced nausea", "CINV",
    ],
    "vomiting": [
        "vomiting", "emesis", "retching", "throwing up",
        "chemotherapy-induced vomiting",
    ],
    "fever": [
        "fever", "pyrexia", "febrile", "high temperature",
        "elevated temperature", "hyperthermia",
    ],
    "neuropathy": [
        "peripheral neuropathy", "neuropathy", "paresthesia", "paresthesias",
        "numbness", "tingling", "pins and needles",
        "burning sensation in hands", "burning sensation in feet",
        "chemotherapy-induced neuropathy", "CIPN",
        "sensory neuropathy", "motor neuropathy",
    ],
    "dyspnea": [
        "dyspnea", "shortness of breath", "SOB", "breathlessness",
        "difficulty breathing", "respiratory distress", "air hunger",
        "exertional dyspnea",
    ],
    "pain": [
        "pain", "painful", "aching", "burning pain", "nociceptive pain",
        "neuropathic pain", "cancer pain", "treatment-related pain",
    ],
    "bleeding": [
        "bleeding", "hemorrhage", "haemorrhage", "hematuria", "haematuria",
        "hemoptysis", "haemoptysis", "epistaxis", "bruising", "ecchymosis",
        "petechiae", "purpura", "blood in urine",
    ],
    "infection": [
        "infection", "sepsis", "febrile neutropenia", "bacteremia",
        "bacteraemia", "cellulitis", "pneumonia", "urinary tract infection",
        "fungal infection", "opportunistic infection",
    ],
    "thrombosis": [
        "thrombosis", "deep vein thrombosis", "DVT", "pulmonary embolism", "PE",
        "venous thromboembolism", "VTE", "blood clot", "thrombus", "clot",
    ],
    "pneumonitis": [
        "pneumonitis", "checkpoint pneumonitis", "immune-related pneumonitis",
        "interstitial lung disease", "ILD", "radiation pneumonitis",
        "drug-induced pneumonitis",
    ],
    "colitis": [
        "colitis", "immune-related colitis", "checkpoint colitis",
        "immunotherapy-related colitis", "diarrhea", "diarrhoea",
        "loose stools", "watery stools",
    ],
    "hepatotoxicity": [
        "hepatotoxicity", "elevated liver enzymes", "transaminase elevation",
        "elevated ALT", "elevated AST", "hepatitis", "jaundice",
        "drug-induced hepatitis", "immune-related hepatitis",
    ],
    "mucositis": [
        "mucositis", "oral mucositis", "stomatitis", "mouth sores",
        "mouth ulcers", "mucosal inflammation", "oral ulceration",
    ],
    "alopecia": [
        "alopecia", "hair loss", "hair thinning", "chemotherapy-induced alopecia",
    ],
    "rash": [
        "rash", "skin rash", "dermatitis", "pruritus", "itching",
        "urticaria", "hives", "maculopapular rash", "immune-related rash",
        "checkpoint rash",
    ],
    "neutropenia": [
        "neutropenia", "low neutrophils", "febrile neutropenia",
        "grade 3 neutropenia", "grade 4 neutropenia",
        "neutrophil count low",
    ],
    "anemia": [
        "anemia", "anaemia", "low hemoglobin", "low haemoglobin",
        "low Hgb", "hemoglobin low", "treatment-related anemia",
    ],
    "thrombocytopenia": [
        "thrombocytopenia", "low platelets", "platelet count low",
        "platelet count reduced", "thrombocytopenic",
    ],
}

# CTCAE grade keywords
_GRADE_PATTERNS = {
    1: r"\b(mild|grade[ -]?1|ctcae[ -]?grade[ -]?1)\b",
    2: r"\b(moderate|grade[ -]?2|ctcae[ -]?grade[ -]?2)\b",
    3: r"\b(severe|grade[ -]?3|ctcae[ -]?grade[ -]?3)\b",
    4: r"\b(life[ -]?threatening|grade[ -]?4|ctcae[ -]?grade[ -]?4)\b",
}
_GRADE_RE = {g: re.compile(pat, re.IGNORECASE) for g, pat in _GRADE_PATTERNS.items()}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class OncologyMention:
    """
    A symptom or adverse event mention detected in a clinical note.

    Attributes:
        text:     Matched text span as it appears in the note.
        category: Symptom category (e.g. "fatigue", "neuropathy").
        start:    Character offset of span start.
        end:      Character offset of span end.
        negated:  True if a negation cue was detected.
        grade:    CTCAE grade 1–4 if grade language found in context window.
                  None if no grade language detected.
    """

    text: str
    category: str
    start: int
    end: int
    negated: bool = False
    grade: Optional[int] = None


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class OncologyExtractor:
    """
    Rule-based oncology symptom and adverse event extractor.

    Uses spaCy PhraseMatcher with a curated oncology vocabulary.
    Optionally annotates mentions with negation and CTCAE grade.
    """

    def __init__(
        self,
        extra_vocabulary: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Build the PhraseMatcher from the curated oncology vocabulary.

        Args:
            extra_vocabulary: Optional additional {category: [synonyms]} dict
                              to merge into the default vocabulary.
        """
        self._nlp = spacy.blank("en")
        self._matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")

        vocab = dict(ONCOLOGY_VOCABULARY)
        if extra_vocabulary:
            for cat, terms in extra_vocabulary.items():
                vocab.setdefault(cat, []).extend(terms)

        # Map match_id (hash) → category
        self._id_to_category: Dict[int, str] = {}

        for category, synonyms in vocab.items():
            patterns = []
            for term in synonyms:
                if term and term.strip():
                    patterns.append(self._nlp.make_doc(term))
            if patterns:
                self._matcher.add(category, patterns)
                # Store the vocab string hash → category
                self._id_to_category[self._nlp.vocab.strings[category]] = category

    def extract(
        self,
        clinical_note: str,
        negation_handler=None,
    ) -> List[OncologyMention]:
        """
        Run the PhraseMatcher on the clinical note.

        Overlapping spans are resolved with spaCy's longest-wins filter.
        Optionally annotates mentions with negation (requires NegationHandler).
        CTCAE grade is inferred from a ±150-character context window.

        Args:
            clinical_note:    Free-text clinical note to process.
            negation_handler: Optional NegationHandler instance for negation.

        Returns:
            List of OncologyMention objects (empty if no mentions found).
        """
        if not clinical_note:
            return []

        doc = self._nlp(clinical_note)
        raw_matches = self._matcher(doc)

        # Group by (category, span)
        cat_spans: Dict[str, list] = {}
        for match_id, start, end in raw_matches:
            span = doc[start:end]
            category = self._id_to_category.get(match_id, "unknown")
            cat_spans.setdefault(category, []).append(span)

        # Resolve overlaps per category, then across all
        all_spans = []
        for category, spans in cat_spans.items():
            filtered = spacy.util.filter_spans(spans)
            for span in filtered:
                all_spans.append((category, span))

        # Re-filter across all categories to handle cross-category overlaps
        span_objs = [s for _, s in all_spans]
        filtered_span_objs = spacy.util.filter_spans(span_objs)
        filtered_set = {(s.start_char, s.end_char) for s in filtered_span_objs}

        mentions: List[OncologyMention] = []
        for category, span in all_spans:
            if (span.start_char, span.end_char) not in filtered_set:
                continue
            grade = self._infer_grade(clinical_note, span.start_char, span.end_char)
            mentions.append(
                OncologyMention(
                    text=span.text,
                    category=category,
                    start=span.start_char,
                    end=span.end_char,
                    grade=grade,
                )
            )

        # Apply negation if handler provided
        if negation_handler is not None and mentions:
            from src.extraction.extractor import Mention as HPOMention
            hpo_mentions = [
                HPOMention(text=m.text, start=m.start, end=m.end)
                for m in mentions
            ]
            annotated = negation_handler.annotate(doc, hpo_mentions)
            for onc_m, hpo_m in zip(mentions, annotated):
                onc_m.negated = hpo_m.negated

        return mentions

    def _infer_grade(self, text: str, start: int, end: int, window: int = 150) -> Optional[int]:
        """
        Scan a ±window character context around the span for CTCAE grade language.

        Returns the highest grade found, or None.
        """
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        context = text[ctx_start:ctx_end]

        for grade in (4, 3, 2, 1):
            if _GRADE_RE[grade].search(context):
                return grade
        return None

    def extract_structured(
        self,
        clinical_note: str,
        negation_handler=None,
    ) -> Dict[str, object]:
        """
        Return a structured dict of per-category binary flags and mention lists.

        Useful for feeding into the risk prediction feature matrix.

        Returns a dict with:
          - "mentions": list of OncologyMention
          - "categories_present": set of category names (non-negated only)
          - "has_<category>": bool for each category in ONCOLOGY_VOCABULARY
          - "symptom_count": count of distinct non-negated categories
          - "max_grade": highest CTCAE grade among non-negated mentions, or None
        """
        mentions = self.extract(clinical_note, negation_handler=negation_handler)
        non_neg = [m for m in mentions if not m.negated]

        present = {m.category for m in non_neg}
        grades = [m.grade for m in non_neg if m.grade is not None]

        result: Dict[str, object] = {
            "mentions": mentions,
            "categories_present": present,
            "symptom_count": len(present),
            "max_grade": max(grades) if grades else None,
        }

        for cat in ONCOLOGY_VOCABULARY:
            result[f"has_{cat}"] = cat in present

        return result
