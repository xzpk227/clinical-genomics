"""
Snakemake pipeline for the Clinical Phenotype Extraction and HPO Mapping Pipeline.

Usage:
    # Build all data artifacts and run tests
    snakemake --cores 1

    # Build only the data artifacts
    snakemake build_data --cores 1

    # Run tests only (assumes data artifacts exist)
    snakemake test --cores 1

    # Start the API server
    snakemake serve --cores 1

    # Run with Docker (recommended)
    snakemake --cores 1 --use-singularity
"""

# Configuration
HPO_SOURCE = "data/hp.json"
HPO_DATABASE = "data/hpo_database.json"
FAISS_INDEX = "data/hpo_index.faiss"
FAISS_ID_MAP = "data/hpo_id_map.json"
EMBEDDING_MODEL = "FremyCompany/BioLORD-2023"
DOCKER_IMAGE = "clinical-phenotype-pipeline-test"


rule all:
    """Default target: build data artifacts and run tests."""
    input:
        HPO_DATABASE,
        FAISS_INDEX,
        FAISS_ID_MAP,
        "results/tests_passed.txt",


# ---------------------------------------------------------------------------
# Data artifact build rules
# ---------------------------------------------------------------------------


rule build_hpo_database:
    """Parse hp.json and build the structured HPO database."""
    input:
        HPO_SOURCE,
    output:
        HPO_DATABASE,
    shell:
        """
        docker run --rm \
            -v "$(pwd)/data:/app/data" \
            {DOCKER_IMAGE} \
            python -c "
from src.data.build_hpo_db import build_hpo_database
db = build_hpo_database('{input}', '{output}')
print(f'HPO database built: version={{db.version}}, terms={{len(db.terms)}}')
"
        """


rule build_faiss_index:
    """Encode HPO terms with BioLORD-2023 and build the FAISS index."""
    input:
        HPO_DATABASE,
    output:
        index=FAISS_INDEX,
        id_map=FAISS_ID_MAP,
    shell:
        """
        docker run --rm \
            -v "$(pwd)/data:/app/data" \
            -v "$(pwd)/.cache:/root/.cache" \
            {DOCKER_IMAGE} \
            python -u -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from sentence_transformers import SentenceTransformer
from src.data.build_hpo_db import load_hpo_database
from src.mapping.build_index import build_faiss_index
print('Loading HPO database...', flush=True)
hpo_db = load_hpo_database('{input}')
print(f'Database loaded: {{len(hpo_db.terms)}} terms', flush=True)
print('Loading embedding model...', flush=True)
model = SentenceTransformer('{EMBEDDING_MODEL}')
print('Model loaded. Building FAISS index...', flush=True)
build_faiss_index(hpo_db, model, '{output.index}', '{output.id_map}')
print('FAISS index built.', flush=True)
"
        """


rule build_data:
    """Build all data artifacts (HPO database + FAISS index)."""
    input:
        HPO_DATABASE,
        FAISS_INDEX,
        FAISS_ID_MAP,


# ---------------------------------------------------------------------------
# Docker image build
# ---------------------------------------------------------------------------


rule build_docker:
    """Build the Docker test image."""
    input:
        "Dockerfile",
        "pyproject.toml",
    output:
        touch("results/docker_built.txt"),
    shell:
        """
        docker build --target test -t {DOCKER_IMAGE} .
        """


# ---------------------------------------------------------------------------
# Test rules
# ---------------------------------------------------------------------------


rule test_unit:
    """Run unit tests inside Docker."""
    input:
        "results/docker_built.txt",
    output:
        touch("results/unit_tests_passed.txt"),
    shell:
        """
        docker run --rm {DOCKER_IMAGE} \
            pytest tests/unit/ -v --tb=short
        """


rule test_integration:
    """Run integration tests inside Docker."""
    input:
        "results/docker_built.txt",
    output:
        touch("results/integration_tests_passed.txt"),
    shell:
        """
        docker run --rm {DOCKER_IMAGE} \
            pytest tests/integration/ -v --tb=short
        """


rule test_regression:
    """Run regression test (requires data artifacts)."""
    input:
        "results/docker_built.txt",
        HPO_DATABASE,
        FAISS_INDEX,
        FAISS_ID_MAP,
    output:
        touch("results/regression_tests_passed.txt"),
    shell:
        """
        docker run --rm \
            -v "$(pwd)/data:/app/data" \
            -v "$(pwd)/.cache:/root/.cache" \
            {DOCKER_IMAGE} \
            pytest tests/regression/ -v --tb=short
        """


rule test:
    """Run all tests (unit + integration + regression)."""
    input:
        "results/unit_tests_passed.txt",
        "results/integration_tests_passed.txt",
        "results/regression_tests_passed.txt",
    output:
        touch("results/tests_passed.txt"),


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


rule evaluate:
    """Run the evaluation suite and produce a metrics report."""
    input:
        HPO_DATABASE,
        FAISS_INDEX,
        FAISS_ID_MAP,
        "results/docker_built.txt",
    output:
        "data/evaluation/report.json",
    shell:
        """
        docker run --rm \
            -v "$(pwd)/data:/app/data" \
            -v "$(pwd)/.cache:/root/.cache" \
            {DOCKER_IMAGE} \
            python -c "
from src.pipeline import Pipeline, PipelineConfig
from src.evaluation.evaluator import Evaluator
pipeline = Pipeline(PipelineConfig())
evaluator = Evaluator()
result = evaluator.run(pipeline)
evaluator.save_report(result, '{output}')
print(result)
"
        """


# ---------------------------------------------------------------------------
# Serve
# ---------------------------------------------------------------------------


rule serve:
    """Start the API server via docker-compose."""
    input:
        HPO_DATABASE,
        FAISS_INDEX,
        FAISS_ID_MAP,
    shell:
        """
        docker compose up --build
        """


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


rule clean:
    """Remove all generated artifacts."""
    shell:
        """
        rm -f {HPO_DATABASE} {FAISS_INDEX} {FAISS_ID_MAP}
        rm -rf results/
        rm -f data/evaluation/report.json
        """
