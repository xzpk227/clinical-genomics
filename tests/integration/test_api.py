"""
Integration tests for the FastAPI endpoints.

Uses FastAPI's TestClient with a stub pipeline injected via app.state —
no real models are loaded.

Requirements covered: 1.4, 1.5, 6.1, 6.2, 6.3, 6.4, 6.6, 9.1, 9.2, 9.3, 12.3
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.extraction.extractor import Mention
from src.mapping.mapper import HPOCandidate
from src.pipeline import ExtractionResult

DISCLAIMER = (
    "This output is for clinical decision support only. "
    "It does not constitute a medical diagnosis. "
    "Always involve a qualified clinician."
)


# ---------------------------------------------------------------------------
# Stub pipeline
# ---------------------------------------------------------------------------


class _StubPipeline:
    """Minimal pipeline stub that returns a fixed ExtractionResult."""

    def __init__(
        self,
        mentions: list[Mention] | None = None,
        mappings: dict | None = None,
        summary: str | None = None,
    ) -> None:
        self.is_ready = True
        self._mentions = mentions or []
        self._mappings = mappings or {}
        self._summary = summary

    def process(self, clinical_note: str, top_k: int = 3) -> ExtractionResult:
        return ExtractionResult(
            mentions=self._mentions,
            mappings=self._mappings,
            summary=self._summary,
        )


def _default_stub() -> _StubPipeline:
    """Stub that returns one seizure mention with one HPO candidate."""
    mention = Mention(text="seizures", start=12, end=20, negated=False)
    candidate = HPOCandidate(hpo_id="HP:0001250", hpo_label="Seizure", confidence=0.95)
    return _StubPipeline(
        mentions=[mention],
        mappings={"seizures": [candidate]},
    )


# ---------------------------------------------------------------------------
# Fixture: TestClient with stub pipeline pre-injected
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """TestClient with a ready stub pipeline in app.state."""
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.pipeline = _default_stub()
        app.state.pipeline_ready = True
        yield c


@pytest.fixture()
def client_not_ready() -> TestClient:
    """TestClient with pipeline_ready=False."""
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.pipeline = None
        app.state.pipeline_ready = False
        yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200_when_ready(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_returns_503_when_not_ready(
        self, client_not_ready: TestClient
    ) -> None:
        resp = client_not_ready.get("/health")
        assert resp.status_code == 503
        assert resp.json()["status"] == "unavailable"


# ---------------------------------------------------------------------------
# POST /extract-phenotypes — success cases
# ---------------------------------------------------------------------------


class TestExtractPhenotypesSuccess:
    def test_extract_returns_200_with_valid_note(self, client: TestClient) -> None:
        resp = client.post(
            "/extract-phenotypes",
            json={"clinical_note": "Patient has seizures."},
        )
        assert resp.status_code == 200

    def test_response_has_hpo_terms_array(self, client: TestClient) -> None:
        resp = client.post(
            "/extract-phenotypes",
            json={"clinical_note": "Patient has seizures."},
        )
        data = resp.json()
        assert "hpo_terms" in data
        assert isinstance(data["hpo_terms"], list)

    def test_response_includes_disclaimer(self, client: TestClient) -> None:
        resp = client.post(
            "/extract-phenotypes",
            json={"clinical_note": "Patient has seizures."},
        )
        data = resp.json()
        assert "disclaimer" in data
        assert data["disclaimer"] == DISCLAIMER

    def test_hpo_term_has_required_fields(self, client: TestClient) -> None:
        resp = client.post(
            "/extract-phenotypes",
            json={"clinical_note": "Patient has seizures."},
        )
        term = resp.json()["hpo_terms"][0]
        for field in ("text", "hpo_id", "hpo_label", "confidence", "negated", "candidates"):
            assert field in term, f"Missing field: {field}"

    def test_hpo_term_confidence_in_range(self, client: TestClient) -> None:
        resp = client.post(
            "/extract-phenotypes",
            json={"clinical_note": "Patient has seizures."},
        )
        for term in resp.json()["hpo_terms"]:
            assert 0.0 <= term["confidence"] <= 1.0

    def test_hpo_term_negated_is_bool(self, client: TestClient) -> None:
        resp = client.post(
            "/extract-phenotypes",
            json={"clinical_note": "Patient has seizures."},
        )
        for term in resp.json()["hpo_terms"]:
            assert isinstance(term["negated"], bool)

    def test_empty_note_returns_empty_hpo_terms(self, client: TestClient) -> None:
        """When stub returns no mentions, hpo_terms should be empty."""
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.pipeline = _StubPipeline(mentions=[], mappings={})
            app.state.pipeline_ready = True
            resp = c.post(
                "/extract-phenotypes",
                json={"clinical_note": "No findings."},
            )
        assert resp.status_code == 200
        assert resp.json()["hpo_terms"] == []


# ---------------------------------------------------------------------------
# POST /extract-phenotypes — validation errors
# ---------------------------------------------------------------------------


class TestExtractPhenotypesValidation:
    def test_extract_returns_422_for_empty_note(self, client: TestClient) -> None:
        resp = client.post(
            "/extract-phenotypes",
            json={"clinical_note": ""},
        )
        assert resp.status_code == 422

    def test_extract_returns_422_for_oversized_note(self, client: TestClient) -> None:
        resp = client.post(
            "/extract-phenotypes",
            json={"clinical_note": "a" * 10_001},
        )
        assert resp.status_code == 422

    def test_extract_returns_422_for_top_k_zero(self, client: TestClient) -> None:
        resp = client.post(
            "/extract-phenotypes",
            json={"clinical_note": "Patient has seizures.", "top_k": 0},
        )
        assert resp.status_code == 422

    def test_extract_returns_422_for_top_k_eleven(self, client: TestClient) -> None:
        resp = client.post(
            "/extract-phenotypes",
            json={"clinical_note": "Patient has seizures.", "top_k": 11},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /extract-phenotypes — pipeline not ready
# ---------------------------------------------------------------------------


class TestExtractPhenotypesNotReady:
    def test_extract_returns_503_when_pipeline_not_ready(
        self, client_not_ready: TestClient
    ) -> None:
        resp = client_not_ready.post(
            "/extract-phenotypes",
            json={"clinical_note": "Patient has seizures."},
        )
        assert resp.status_code == 503
        assert "detail" in resp.json()


# ---------------------------------------------------------------------------
# POST /extract-phenotypes — negated mention
# ---------------------------------------------------------------------------


class TestNegatedMentionInResponse:
    def test_negated_mention_in_response(self, client: TestClient) -> None:
        """A negated mention must appear in hpo_terms with negated=True."""
        mention = Mention(text="hearing loss", start=3, end=15, negated=True)
        candidate = HPOCandidate(
            hpo_id="HP:0000365", hpo_label="Hearing impairment", confidence=0.88
        )
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.pipeline = _StubPipeline(
                mentions=[mention],
                mappings={"hearing loss": [candidate]},
            )
            app.state.pipeline_ready = True
            resp = c.post(
                "/extract-phenotypes",
                json={"clinical_note": "No hearing loss noted."},
            )

        assert resp.status_code == 200
        terms = resp.json()["hpo_terms"]
        assert len(terms) == 1
        assert terms[0]["negated"] is True
        assert terms[0]["text"] == "hearing loss"

    def test_negated_mention_still_has_candidates(self, client: TestClient) -> None:
        """A negated mention must still include HPO candidates (Requirement 3.5)."""
        mention = Mention(text="hearing loss", start=3, end=15, negated=True)
        candidate = HPOCandidate(
            hpo_id="HP:0000365", hpo_label="Hearing impairment", confidence=0.88
        )
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.pipeline = _StubPipeline(
                mentions=[mention],
                mappings={"hearing loss": [candidate]},
            )
            app.state.pipeline_ready = True
            resp = c.post(
                "/extract-phenotypes",
                json={"clinical_note": "No hearing loss noted."},
            )

        term = resp.json()["hpo_terms"][0]
        assert len(term["candidates"]) > 0
        assert term["candidates"][0]["hpo_id"] == "HP:0000365"
