"""
Opus 4.7 narrative translator.

Given a previously-produced ``CausalEvidenceEnvelope`` (from any other
endpoint), asks Claude to produce a plain-English paragraph the calling
agent can cite verbatim.

Fallback
--------
If ``ANTHROPIC_API_KEY`` is unset, ``explain_causal_evidence`` returns a
deterministic template-generated paragraph instead of calling out. This
keeps the service runnable in fully-offline contexts (CI, air-gapped
demos) at the cost of a blander narrative.

Prompt caching
--------------
The system prompt is large + stable, so we mark it ``cache_control:
ephemeral`` per Anthropic's caching docs. Cache hits save ~80% of the
input-token cost on repeated calls, which matters when a client
re-requests narrations frequently for the same dashboard.
"""

from __future__ import annotations

import json
import os

from app.schemas import CausalEvidenceEnvelope, ExplainInput

_ENGINE_VERSION = "anthropic-opus-4-7"

# System prompt deliberately verbose — it establishes vocabulary + constraints
# and is stable across requests, so prompt caching pays off.
SYSTEM_PROMPT = """\
You are the narrative translator for a causal-evidence service.
You receive a structured CausalEvidence envelope — the output of a causal-
inference endpoint (backdoor identification, sensitivity refutation,
Bayesian counterfactual, heterogeneous treatment effect, or DAG inference)
— and produce ONE short paragraph of plain English suitable for an
operator reading a dashboard panel.

Constraints:
- Output 2-4 sentences. No bullets, no headings, no code.
- Cite the numeric effect size and, when present, the credible interval.
  Use the envelope's exact numbers; do not round aggressively.
- Name the evidence level ("strong", "moderate", "preliminary", or
  "insufficient") verbatim.
- Surface the top caveat from the envelope in the final sentence.
- Never speculate beyond what the envelope supports. If evidence_level is
  "insufficient", say the service refused to compute and name the gate.
- Do not mention DoWhy, PyMC, or EconML by name — the audience is
  operators, not statisticians.
"""


async def explain_causal_evidence(inp: ExplainInput) -> CausalEvidenceEnvelope:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    envelope = inp.envelope

    if not api_key:
        return _offline_fallback(envelope, inp.question)

    # Lazy import so offline tests don't need the anthropic package.
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=api_key)
    payload = json.dumps(envelope.model_dump(by_alias=True), indent=2, default=str)
    user_message = (
        f"Operator question: {inp.question}\n\n"
        f"Envelope (JSON):\n```json\n{payload}\n```\n\n"
        "Write the paragraph now."
    )

    model = os.getenv("CLAUDE_EXPLAINER_MODEL", "claude-opus-4-7").strip()
    response = await client.messages.create(
        model=model,
        max_tokens=inp.max_tokens,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    narrative = _concat_text_blocks(response)
    return _narrative_envelope(envelope, narrative, used_llm=True)


def _concat_text_blocks(response: object) -> str:
    content = getattr(response, "content", [])
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts).strip()


def _offline_fallback(
    envelope: CausalEvidenceEnvelope, question: str
) -> CausalEvidenceEnvelope:
    ev = envelope.evidence
    bits = [
        f"Question: {question}",
        (
            f"Evidence level is {ev.evidence_level} over "
            f"{ev.observations} observation{'s' if ev.observations != 1 else ''}."
        ),
        f"Reported effect size is {ev.effect_size:.3f} via {ev.methodology}.",
    ]
    if ev.p_value is not None:
        bits.append(f"p-value ≈ {ev.p_value:.3f}.")
    if ev.confounders:
        bits.append(
            "Unblocked confounders: " + ", ".join(sorted(ev.confounders)) + "."
        )
    bits.append(f"Caveat: {ev.caveat}")
    return _narrative_envelope(envelope, " ".join(bits), used_llm=False)


def _narrative_envelope(
    src: CausalEvidenceEnvelope, narrative: str, *, used_llm: bool
) -> CausalEvidenceEnvelope:
    """Wrap a narrative paragraph back into a CausalEvidenceEnvelope so the
    agent sees a uniform return shape across all endpoints."""
    return CausalEvidenceEnvelope(
        evidence=src.evidence,
        details={
            "narrative": narrative,
            "usedLlm": used_llm,
            "sourceEndpoint": src.endpoint,
        },
        endpoint="/explain/causal-evidence",
        engineVersion=_ENGINE_VERSION,
    )
