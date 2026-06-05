"""
Contract tests — the CausalEvidence envelope must serialize to the exact
camelCase shape consuming agents expect. If this test fails, any agent
tool bindings need to be updated in lockstep.
"""

from __future__ import annotations

from app.schemas import CausalEvidence, CausalEvidenceEnvelope


def test_causal_evidence_serializes_camel_case():
    evidence = CausalEvidence(
        evidenceLevel="moderate",
        observations=42,
        effectSize=0.42,
        pValue=0.01,
        confounders=["weather"],
        methodology="backdoor_criterion",
        caveat="cohort <100",
    )
    serialized = evidence.model_dump(by_alias=True)
    assert serialized == {
        "evidenceLevel": "moderate",
        "observations": 42,
        "effectSize": 0.42,
        "pValue": 0.01,
        "confounders": ["weather"],
        "methodology": "backdoor_criterion",
        "caveat": "cohort <100",
    }


def test_envelope_round_trips_with_aliases():
    envelope = CausalEvidenceEnvelope(
        evidence=CausalEvidence(
            evidenceLevel="strong",
            observations=300,
            effectSize=1.2,
            pValue=0.0001,
            confounders=[],
            methodology="sensitivity_add_unobserved_common_cause",
            caveat="robust",
        ),
        details={"refutedEffect": 0.85},
        endpoint="/causal/sensitivity",
        engineVersion="dowhy-0.11",
    )
    dumped = envelope.model_dump(by_alias=True)
    assert dumped["endpoint"] == "/causal/sensitivity"
    assert dumped["engineVersion"] == "dowhy-0.11"
    assert dumped["evidence"]["evidenceLevel"] == "strong"


def test_p_value_can_be_null():
    evidence = CausalEvidence(
        evidenceLevel="insufficient",
        observations=5,
        effectSize=0.0,
        confounders=[],
        methodology="anything",
        caveat="too small",
    )
    dumped = evidence.model_dump(by_alias=True)
    assert dumped["pValue"] is None
