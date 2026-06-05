"""
Explain-endpoint fallback tests.

When ``ANTHROPIC_API_KEY`` is unset the offline template kicks in. These
tests assert its shape so operators running air-gapped demos see sensible
output without loading the anthropic SDK.
"""

from __future__ import annotations

import pytest

from app.engines.explain import explain_causal_evidence
from app.schemas import (
    CausalEvidence,
    CausalEvidenceEnvelope,
    ExplainInput,
)


@pytest.fixture(autouse=True)
def _unset_anthropic_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _sample_envelope(p_value: float | None = None) -> CausalEvidenceEnvelope:
    return CausalEvidenceEnvelope(
        evidence=CausalEvidence(
            evidenceLevel="moderate",
            observations=42,
            effectSize=0.65,
            pValue=p_value,
            confounders=["covariate_a", "covariate_b"],
            methodology="backdoor_criterion",
            caveat="two confounders remain unblocked",
        ),
        details={"recommendedAdjustmentSet": ["covariate_a"]},
        endpoint="/causal/backdoor-check",
        engineVersion="dowhy-0.11",
    )


async def test_offline_fallback_cites_numbers_and_caveat():
    result = await explain_causal_evidence(
        ExplainInput(
            envelope=_sample_envelope(p_value=0.03),
            question="Is the operator's time savings real?",
        )
    )
    narrative = result.details["narrative"]
    assert "moderate" in narrative
    assert "42" in narrative
    assert "0.65" in narrative.replace("0.650", "0.65")
    assert "two confounders remain unblocked" in narrative
    # Provenance: tells the agent it's the fallback, not the LLM.
    assert result.details["usedLlm"] is False
    assert result.details["sourceEndpoint"] == "/causal/backdoor-check"


async def test_offline_fallback_handles_missing_p_value():
    result = await explain_causal_evidence(
        ExplainInput(
            envelope=_sample_envelope(p_value=None),
            question="Any confidence?",
        )
    )
    narrative = result.details["narrative"]
    # Does not fabricate a p-value when one isn't supplied.
    assert "p-value" not in narrative.lower() or "none" in narrative.lower()
