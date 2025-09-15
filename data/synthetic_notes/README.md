# Synthetic Clinical Notes

This directory contains synthetic, de-identified clinical note examples for testing and demonstration purposes only.

## Important Notice

**These notes contain NO real patient data.** All clinical scenarios, patient details, names, dates, and findings are entirely fictitious and have been artificially constructed for the purpose of testing and demonstrating the Clinical Phenotype Extraction and HPO Mapping Pipeline.

These notes must not be used for any clinical decision-making, research involving real patients, or any purpose other than software testing and demonstration.

## Contents

| File | Description |
|------|-------------|
| `note_01_seizures_hypotonia.txt` | Pediatric patient with seizures, hypotonia, and global developmental delay |
| `note_02_negated_findings.txt` | Patient with autism spectrum disorder; multiple negated phenotypes (no seizures, without hypotonia, denies hearing loss) |
| `note_03_multi_phenotype.txt` | Infant with multiple congenital anomalies: microcephaly, short stature, dysmorphic features, hypotonia, absent speech |
| `note_04_neuromuscular.txt` | Adolescent with Duchenne muscular dystrophy: muscle weakness, scoliosis, cardiomyopathy; negated intellectual disability |
| `note_05_metabolic_disorder.txt` | Infant with suspected MCAD deficiency: hypoglycemia, hepatomegaly, hypotonia |

## Purpose

These notes are used to:
1. Demonstrate the pipeline's ability to extract phenotype mentions from realistic clinical language
2. Test negation detection (e.g., "no seizures", "without hypotonia", "denies hearing loss")
3. Test multi-phenotype extraction from a single note
4. Validate the pipeline's handling of diverse clinical contexts

## Usage

```bash
# Run the pipeline on a synthetic note via the API
curl -X POST http://localhost:8000/extract-phenotypes \
  -H "Content-Type: application/json" \
  -d "{\"clinical_note\": \"$(cat data/synthetic_notes/note_01_seizures_hypotonia.txt)\"}"
```
