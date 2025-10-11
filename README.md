# Clinical NLP & Oncology Adverse Event Prediction Pipeline

A clinical NLP and machine learning pipeline that converts free-text clinical notes and structured EHR data into actionable oncology decision-support outputs. Built as a portfolio project demonstrating clinical NLP, biomedical ontology mapping, embedding-based retrieval, adverse event risk prediction, and responsible AI practices.

> **This system is decision-support tooling only. It does not diagnose patients and must always be used under clinician supervision.**

---

## Overview

The pipeline has two integrated components:

| Component | Description |
|-----------|-------------|
| **HPO Extraction Pipeline** | Converts free-text clinical notes into structured [Human Phenotype Ontology (HPO)](https://hpo.jax.org/) term mappings using spaCy, negspaCy/NegEx, BioLORD-2023 embeddings, and FAISS semantic search |
| **Oncology Adverse Event Module** | Extracts oncology symptoms and CTCAE-graded adverse events from clinical notes; predicts adverse event risk from structured EHR features using a calibrated XGBoost + Logistic Regression ensemble |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          FastAPI Service                              │
│                                                                      │
│  POST /extract-phenotypes          HPO term extraction               │
│  POST /extract-clinical-concepts   Oncology symptom extraction       │
│  POST /predict-adverse-event-risk  AE risk prediction                │
│  POST /summarize-risk              LLM-generated risk summary        │
│  GET  /health                                                        │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
           ▼                           ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│   HPO Pipeline        │   │   Oncology Pipeline           │
│                      │   │                              │
│  [Extractor]         │   │  [OncologyExtractor]         │
│  spaCy PhraseMatcher │   │  16-category vocabulary      │
│                      │   │  CTCAE grade inference       │
│  [Negation Handler]  │   │                              │
│  negspaCy NegEx      │   │  [NegationHandler]           │
│                      │   │  negspaCy NegEx              │
│  [Mapper]            │   │                              │
│  BioLORD-2023        │   │  [AdverseEventModel]         │
│  FAISS IndexFlatIP   │   │  XGBoost + LR ensemble       │
│                      │   │  Calibrated probabilities    │
│  [LLM Summary]       │   │                              │
│  (feature-flagged)   │   │  [LLM Risk Summary]          │
└──────────────────────┘   │  (feature-flagged)           │
                           └──────────────────────────────┘
```

### HPO Pipeline Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Extractor** | spaCy `PhraseMatcher` | Detects HPO term mentions via case-insensitive rule-based matching |
| **Negation Handler** | negspaCy (NegEx) | Identifies negated mentions using a configurable cue list |
| **Mapper** | BioLORD-2023 + FAISS | Encodes mentions into biomedical vector space; retrieves top-k HPO candidates by cosine similarity |
| **LLM Summary** | Local medical LLM (optional) | Generates plain-language summaries from structured HPO results |

### Oncology Module Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **OncologyExtractor** | spaCy `PhraseMatcher` | Detects 18 oncology symptom categories with CTCAE grade inference (grades 1–4) |
| **AdverseEventModel** | XGBoost + Logistic Regression | Predicts adverse event probability from EHR features + NLP-derived symptom flags |
| **Feature Engineering** | scikit-learn `ColumnTransformer` | Combines lab values, ECOG score, staging, cycle number, and NLP binary flags |

---

## Oncology Adverse Event Model

### Feature Groups

| Group | Features |
|-------|---------|
| **Demographic** | age, sex |
| **Clinical** | ECOG score, cancer stage (I–IV), treatment cycle number |
| **Laboratory** | WBC, ANC, hemoglobin, platelets, creatinine, ALT, bilirubin |
| **Symptom scores** | fatigue, nausea, pain, dyspnea (0–10 scale) |
| **NLP flags** | has_fatigue, has_nausea, has_fever, has_neuropathy, has_dyspnea, has_pain, has_bleeding, has_infection, has_thrombosis, has_pneumonitis, has_colitis, has_hepatotoxicity, has_mucositis, has_neutropenia, has_anemia, has_thrombocytopenia, symptom_count |

### Model Architecture

- **Baseline**: Logistic Regression with L2 regularization, Platt-scaled calibration
- **Primary**: XGBoost gradient boosted trees, isotonic calibration
- **Ensemble**: Soft-vote average of LR and XGBoost probabilities
- **Evaluation**: AUROC, AUPRC (primary for class imbalance), Brier score, subgroup AUROC by age group / sex / cancer type, 5-fold stratified CV

### Oncology Symptom Categories

The extractor covers 18 CTCAE-relevant categories:

`fatigue` · `nausea` · `vomiting` · `fever` · `neuropathy` · `dyspnea` · `pain` · `bleeding` · `infection` · `thrombosis` · `pneumonitis` · `colitis` · `hepatotoxicity` · `mucositis` · `alopecia` · `rash` · `neutropenia` · `anemia` · `thrombocytopenia`

---

## Evaluation Results

Evaluated on 24 synthetic clinical notes covering exact label matches, synonym matches, negated phenotypes, multi-phenotype notes, and edge cases.

| Metric | Score |
|--------|-------|
| Extraction precision | **0.920** |
| Extraction recall | **0.852** |
| Extraction F1 | **0.885** |
| Top-1 HPO accuracy | **0.913** |
| Top-3 HPO accuracy | **1.000** |
| Negation FP rate | **0.000** |

Run the evaluation locally:

```bash
TOKENIZERS_PARALLELISM=false python scripts/run_evaluation.py
```

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (recommended for the API)
- HPO source file: download `hp.json` from the [HPO GitHub releases](https://github.com/obophenotype/human-phenotype-ontology/releases)

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Build the HPO database

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

### 5. Run locally

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## Example API Calls

### Extract HPO phenotypes

```bash
curl -X POST http://localhost:8000/extract-phenotypes \
  -H "Content-Type: application/json" \
  -d '{
    "clinical_note": "Patient has seizures and hypotonia. No hearing loss was detected.",
    "top_k": 3
  }'
```

### Extract oncology symptoms

```bash
curl -X POST http://localhost:8000/extract-clinical-concepts \
  -H "Content-Type: application/json" \
  -d '{
    "clinical_note": "Patient reports severe fatigue and grade 2 peripheral neuropathy. No nausea."
  }'
```

**Response:**
```json
{
  "mentions": [
    {"text": "fatigue", "category": "fatigue", "negated": false, "grade": 3},
    {"text": "peripheral neuropathy", "category": "neuropathy", "negated": false, "grade": 2},
    {"text": "nausea", "category": "nausea", "negated": true, "grade": null}
  ],
  "categories_present": ["fatigue", "neuropathy"],
  "symptom_count": 2,
  "max_grade": 3
}
```

### Predict adverse event risk

```bash
curl -X POST http://localhost:8000/predict-adverse-event-risk \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "age": 62,
    "sex": "F",
    "cancer_type": "breast",
    "stage": "III",
    "treatment": "AC-T",
    "cycle_number": 3,
    "ecog_score": 1,
    "wbc": 3.2, "anc": 1.8, "hemoglobin": 10.5,
    "platelets": 180, "creatinine": 0.9, "alt": 28, "bilirubin": 0.6,
    "fatigue_score": 6.0, "nausea_score": 3.0, "pain_score": 2.0, "dyspnea_score": 1.0
  }'
```

**Response:**
```json
{
  "patient_id": "P001",
  "risk_probability": 0.74,
  "risk_tier": "high",
  "model_used": "xgboost"
}
```

### OpenAPI documentation

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Running Tests

```bash
# Unit tests
docker run --rm clinical-phenotype-pipeline pytest tests/unit/ -v

# Integration tests
docker run --rm clinical-phenotype-pipeline pytest tests/integration/ -v

# Regression tests
docker run --rm clinical-phenotype-pipeline pytest tests/regression/ -v
```

---

## Configuration

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

This system is designed for use with **synthetic or de-identified clinical text only**. No real patient data should be used as input. The pipeline does not store or log clinical note content after the response is returned.

### Clinician-in-the-Loop Requirement

All outputs must be reviewed by a qualified clinician before any clinical action is taken. This pipeline surfaces candidate HPO terms and risk scores for clinician review — it is not a replacement for clinical judgment.

### Confidence Score Interpretation

Confidence scores represent cosine similarity between mention and HPO term embeddings, normalized to [0.0, 1.0]:

- **≥ 0.90**: High confidence — strong semantic match
- **0.70–0.89**: Moderate confidence — likely correct, warrants review
- **< 0.70**: Low confidence — manual verification recommended

### Known Limitations

- **Rule-based extraction**: Only detects spans matching HPO labels or synonyms exactly. Novel phrasings not in the HPO synonym list will be missed.
- **Embedding model scope**: BioLORD-2023 is trained on biomedical literature; performance may vary on colloquial clinical language.
- **Negation scope**: NegEx uses a fixed token window; complex negation patterns may not be detected.
- **No temporal reasoning**: Does not distinguish current, historical, or family history findings.
- **Synthetic data only**: The adverse event model and evaluation are trained and tested on synthetic EHR data. Performance on real patient cohorts requires separate validation.
- **English only**: Designed for English-language clinical notes.

### Prohibition on Diagnostic Use

**This system must not be used as a diagnostic tool.** Every response includes the disclaimer:

> *"This output is for clinical decision support only. It does not constitute a medical diagnosis. Always involve a qualified clinician."*
