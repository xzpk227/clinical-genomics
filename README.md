# Clinical Phenotype Extraction and HPO Mapping Pipeline

A production-quality NLP pipeline that accepts free-text clinical notes and returns structured [Human Phenotype Ontology (HPO)](https://hpo.jax.org/) term mappings. Built as a portfolio project demonstrating clinical NLP, biomedical ontology mapping, embedding-based retrieval, FastAPI deployment, and responsible AI practices.

> **This system is decision-support tooling only. It does not diagnose patients and must always be used under clinician supervision.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Service                          │
│   POST /extract-phenotypes        GET /health               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Pipeline Orchestrator                       │
│                                                             │
│  clinical_note                                              │
│      │                                                      │
│      ▼                                                      │
│  [Extractor]          spaCy PhraseMatcher over HPO terms    │
│      │                                                      │
│      ▼                                                      │
│  [Negation Handler]   negspaCy NegEx algorithm              │
│      │                                                      │
│      ▼                                                      │
│  [Mapper]             BioLORD-2023 + FAISS IndexFlatIP      │
│      │                                                      │
│      ▼  (optional)                                          │
│  [LLM Summary]        Local medical LLM (feature-flagged)   │
└─────────────────────────────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │       Data Layer        │
          │  hpo_database.json      │
          │  hpo_index.faiss        │
          │  hpo_id_map.json        │
          └─────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Extractor** | spaCy `PhraseMatcher` | Detects HPO term mentions in clinical notes via case-insensitive rule-based matching |
| **Negation Handler** | negspaCy (NegEx) | Identifies negated mentions (e.g., "no seizures") using a configurable cue list |
| **Mapper** | BioLORD-2023 + FAISS | Encodes mentions into biomedical vector space and retrieves top-k HPO candidates by cosine similarity |
| **LLM Summary** | Local medical LLM (optional) | Generates plain-language summaries from structured HPO results (feature-flagged) |
| **API** | FastAPI + Pydantic v2 | Validates inputs, orchestrates the pipeline, returns structured JSON responses |

---

## Setup and Build

### Prerequisites

- Python 3.11+
- Docker (recommended for running the API)
- HPO source file: download `hp.json` from the [HPO GitHub releases](https://github.com/obophenotype/human-phenotype-ontology/releases)

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Build the HPO database

Download `hp.json` from the HPO releases page and place it in `data/`:

```bash
python -c "
from src.data.build_hpo_db import build_hpo_database
build_hpo_database('data/hp.json', 'data/hpo_database.json')
"
```

### 3. Build the FAISS index

```bash
python -c "
from sentence_transformers import SentenceTransformer
from src.data.build_hpo_db import load_hpo_database
from src.mapping.build_index import build_faiss_index

hpo_db = load_hpo_database('data/hpo_database.json')
model = SentenceTransformer('FremyCompany/BioLORD-2023')
build_faiss_index(hpo_db, model, 'data/hpo_index.faiss', 'data/hpo_id_map.json')
"
```

### 4. Run with Docker (recommended)

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

### 5. Run locally (without Docker)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## Example API Calls

### Extract phenotypes from a clinical note

```bash
curl -X POST http://localhost:8000/extract-phenotypes \
  -H "Content-Type: application/json" \
  -d '{
    "clinical_note": "Patient has seizures and hypotonia. No hearing loss was detected.",
    "top_k": 3
  }'
```

**Expected response:**

```json
{
  "hpo_terms": [
    {
      "text": "seizures",
      "hpo_id": "HP:0001250",
      "hpo_label": "Seizure",
      "confidence": 0.97,
      "negated": false,
      "candidates": [
        {"hpo_id": "HP:0001250", "hpo_label": "Seizure", "confidence": 0.97},
        {"hpo_id": "HP:0001251", "hpo_label": "Ataxia", "confidence": 0.61},
        {"hpo_id": "HP:0002353", "hpo_label": "EEG abnormality", "confidence": 0.58}
      ]
    },
    {
      "text": "hypotonia",
      "hpo_id": "HP:0001290",
      "hpo_label": "Hypotonia",
      "confidence": 0.99,
      "negated": false,
      "candidates": [...]
    },
    {
      "text": "hearing loss",
      "hpo_id": "HP:0000365",
      "hpo_label": "Hearing impairment",
      "confidence": 0.95,
      "negated": true,
      "candidates": [...]
    }
  ],
  "summary": null,
  "disclaimer": "This output is for clinical decision support only. It does not constitute a medical diagnosis. Always involve a qualified clinician."
}
```

### Check pipeline health

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

### OpenAPI documentation

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Running Tests

All tests run inside Docker to ensure a consistent, isolated environment:

```bash
# Build the image first
docker build -t clinical-phenotype-pipeline .

# Run unit tests
docker run --rm clinical-phenotype-pipeline pytest tests/unit/ -v

# Run integration tests
docker run --rm clinical-phenotype-pipeline pytest tests/integration/ -v

# Run regression test (requires pre-built data artifacts)
docker run --rm clinical-phenotype-pipeline pytest tests/regression/ -v
```

---

## Evaluation Results

The pipeline is evaluated against a curated test set of 24 synthetic clinical notes covering exact label matches, synonym matches, negated phenotypes, multi-phenotype notes, and edge cases.

| Metric | Description |
|--------|-------------|
| `extraction_precision` | Fraction of predicted mentions that match expected mentions |
| `extraction_recall` | Fraction of expected mentions that were predicted |
| `extraction_f1` | Harmonic mean of precision and recall |
| `top1_accuracy` | Fraction of cases where the correct HPO term is the top-1 result |
| `top3_accuracy` | Fraction of cases where the correct HPO term appears in top-3 |
| `negation_fp_rate` | Fraction of negated mentions incorrectly returned as non-negated |

Run the evaluation:

```bash
python -c "
from src.pipeline import Pipeline, PipelineConfig
from src.evaluation.evaluator import Evaluator

pipeline = Pipeline(PipelineConfig())
evaluator = Evaluator()
result = evaluator.run(pipeline)
evaluator.save_report(result, 'data/evaluation/report.json')
print(result)
"
```

The regression test enforces a minimum top-1 accuracy of **0.70** (configurable via `REGRESSION_ACCURACY_THRESHOLD` env var).

---

## Configuration

All configuration is managed via `PipelineConfig` and can be overridden with environment variables:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `HPO_DATABASE_PATH` | `data/hpo_database.json` | Path to the serialized HPO database |
| `FAISS_INDEX_PATH` | `data/hpo_index.faiss` | Path to the FAISS vector index |
| `ID_MAP_PATH` | `data/hpo_id_map.json` | Path to the FAISS ID map |
| `EMBEDDING_MODEL_NAME` | `FremyCompany/BioLORD-2023` | HuggingFace embedding model |
| `TOP_K_DEFAULT` | `3` | Default number of HPO candidates per mention |
| `LLM_SUMMARY_ENABLED` | `false` | Enable optional LLM summary layer |
| `LLM_MODEL_NAME` | `google/medgemma-4b-it` | LLM model for summaries (when enabled) |
| `REGRESSION_ACCURACY_THRESHOLD` | `0.70` | Minimum top-1 accuracy for regression test |
| `PORT` | `8000` | API server port |

---

## Responsible AI

### No Real Patient Data

This system is designed for use with **synthetic or publicly available de-identified clinical text only**. No real patient data should ever be used as input. The pipeline does not store or log clinical note content after the response is returned.

### Clinician-in-the-Loop Requirement

All outputs from this system must be reviewed by a qualified clinician before any clinical action is taken. The pipeline is a decision-support tool — it surfaces candidate HPO terms for clinician review, not a replacement for clinical judgment.

### Confidence Score Interpretation

Confidence scores represent the cosine similarity between the mention embedding and the HPO term embedding, normalized to [0.0, 1.0]:

- **≥ 0.90**: High confidence — strong semantic match
- **0.70–0.89**: Moderate confidence — likely correct, warrants review
- **< 0.70**: Low confidence — treat with caution; manual verification recommended

Scores below 0.70 should not be used to drive clinical decisions without explicit clinician review.

### Known Limitations

- **Rule-based extraction**: The extractor only detects spans that exactly match HPO labels or synonyms. Novel phrasings not in the HPO synonym list will be missed.
- **Embedding model scope**: BioLORD-2023 is trained on biomedical literature; performance may vary on highly colloquial or non-standard clinical language.
- **Negation scope**: The NegEx algorithm uses a fixed token window. Complex negation patterns (e.g., double negatives, long-range dependencies) may not be detected correctly.
- **No temporal reasoning**: The pipeline does not distinguish between current, historical, or family history findings.
- **English only**: The pipeline is designed for English-language clinical notes only.

### Prohibition on Diagnostic Use

**This system must not be used as a diagnostic tool.** It does not provide diagnoses, treatment recommendations, or clinical decisions. Every response includes the following disclaimer:

> *"This output is for clinical decision support only. It does not constitute a medical diagnosis. Always involve a qualified clinician."*
