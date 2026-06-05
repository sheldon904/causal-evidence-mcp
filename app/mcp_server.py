"""
FastMCP server for the causal-evidence service.

Exposes the same 6 causal endpoints offered by ``app.main``'s FastAPI,
but as MCP tools so a calling agent can register the server directly via
the Anthropic SDK's ``mcp_servers`` parameter. No HTTP hop is inserted:
each MCP tool calls the engine function in-process.

Transport: SSE on port 4001 by default (override via ``MCP_SERVER_PORT``).
Anthropic's remote-MCP support requires an SSE endpoint; FastMCP's
``transport="sse"`` exposes that at ``/sse`` on the configured port.

Security:
    - FastMCP tools are called by Claude with the same context the caller
      provides in the agent loop. We do NOT hit the network from tools
      (except when the operator explicitly calls ``explain-causal-evidence``
      and ``ANTHROPIC_API_KEY`` is set, matching the FastAPI surface).
    - If ``MCP_SERVER_SECRET`` is set, the server rejects SSE connections
      without a matching Bearer token. Unset → open locally; the compose
      healthcheck relies on this.

Run
---

Inside the container (Dockerfile sets ENTRYPOINT_MODE=mcp for this):

    python -m app.mcp_server

Or directly:

    uv run python -m app.mcp_server
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.engines.dag_infer import dag_infer
from app.engines.dowhy_engine import backdoor_check, sensitivity
from app.engines.econml_engine import hte_estimate
from app.engines.explain import explain_causal_evidence
from app.engines.pymc_engine import bayesian_counterfactual
from app.schemas import (
    BackdoorCheckInput,
    BayesianCounterfactualInput,
    CausalEvidenceEnvelope,
    DagInferInput,
    ExplainInput,
    HteInput,
    ObservationSeries,
    SensitivityInput,
)

_DEFAULT_PORT = int(os.getenv("MCP_SERVER_PORT", "4001"))
_DEFAULT_HOST = os.getenv("MCP_SERVER_HOST", "0.0.0.0")

# Advertise the service with an informative name — clients surface it in
# tool-use traces ("calling reasoning.causal-backdoor-check").
mcp = FastMCP(
    "reasoning",
    host=_DEFAULT_HOST,
    port=_DEFAULT_PORT,
)


# Each tool mirrors exactly one REST endpoint and returns the raw
# CausalEvidenceEnvelope as a dict. We return dicts (not Pydantic models)
# because MCP's JSON serializer doesn't walk pydantic field aliases the
# same way Pydantic's model_dump does.

def _envelope_to_dict(env: CausalEvidenceEnvelope) -> dict[str, Any]:
    return env.model_dump(by_alias=True, mode="json")


@mcp.tool(
    name="causal-backdoor-check",
    description=(
        "Run DoWhy identify_effect on a caller-supplied DAG + cohort. "
        "Returns a CausalEvidence envelope with the adjustment set and the "
        "candidate confounders that remain unblocked after adjustment. "
        "Use this when the agent asks 'what confounders are still in play?'"
    ),
)
async def causal_backdoor_check(
    treatment: str,
    outcome: str,
    observed_confounders: list[str],
    candidate_confounders: list[str],
    graph_edges: list[tuple[str, str]],
    cohort: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    inp = BackdoorCheckInput(
        treatment=treatment,
        outcome=outcome,
        observedConfounders=observed_confounders,
        candidateConfounders=candidate_confounders,
        graphEdges=graph_edges,
        cohort=[ObservationSeries(**c) for c in (cohort or [])],
    )
    return await asyncio.to_thread(lambda: _envelope_to_dict(backdoor_check(inp)))


@mcp.tool(
    name="causal-sensitivity",
    description=(
        "DoWhy refute_estimate with add_unobserved_common_cause. Returns "
        "the fraction of a hidden confounder's effect strength that would "
        "flip the sign of the observed effect. Use when the agent asks "
        "'how fragile is this conclusion?'"
    ),
)
async def causal_sensitivity(
    treatment: str,
    outcome: str,
    graph_edges: list[tuple[str, str]],
    cohort_treated: list[float],
    cohort_untreated: list[float],
    effect_strength_on_treatment: float = 0.1,
    effect_strength_on_outcome: float = 0.1,
) -> dict[str, Any]:
    inp = SensitivityInput(
        treatment=treatment,
        outcome=outcome,
        graphEdges=graph_edges,
        cohortTreated=cohort_treated,
        cohortUntreated=cohort_untreated,
        effectStrengthOnTreatment=effect_strength_on_treatment,
        effectStrengthOnOutcome=effect_strength_on_outcome,
    )
    return await asyncio.to_thread(lambda: _envelope_to_dict(sensitivity(inp)))


@mcp.tool(
    name="causal-bayesian-counterfactual",
    description=(
        "PyMC NUTS sampler over a two-arm Normal model. Returns posterior "
        "mean + 94% credible interval + ESS + R-hat. Use when the agent "
        "needs a 'how confident' statement, not a point estimate."
    ),
)
async def causal_bayesian_counterfactual(
    treated: list[float],
    untreated: list[float],
    credible_mass: float = 0.94,
    draws: int = 2000,
    tune: int = 1000,
    min_per_arm: int = 10,
) -> dict[str, Any]:
    inp = BayesianCounterfactualInput(
        treated=treated,
        untreated=untreated,
        credibleMass=credible_mass,
        draws=draws,
        tune=tune,
        minPerArm=min_per_arm,
    )
    return await asyncio.to_thread(
        lambda: _envelope_to_dict(bayesian_counterfactual(inp))
    )


@mcp.tool(
    name="causal-hte-estimate",
    description=(
        "EconML CausalForestDML for heterogeneous treatment effects. "
        "Returns per-feature median-split strata (effect below / effect "
        "above). Use when the agent asks 'does this effect vary by "
        "context?'"
    ),
)
async def causal_hte_estimate(
    treatment: list[int],
    outcome: list[float],
    features: list[list[float]],
    feature_names: list[str],
    n_estimators: int = 100,
) -> dict[str, Any]:
    inp = HteInput(
        treatment=treatment,
        outcome=outcome,
        features=features,
        featureNames=feature_names,
        nEstimators=n_estimators,
    )
    return await asyncio.to_thread(lambda: _envelope_to_dict(hte_estimate(inp)))


@mcp.tool(
    name="causal-dag-infer",
    description=(
        "PC algorithm DAG inference over observed variables. Returns an "
        "edge list with bootstrap-based confidence scores. Use when the "
        "agent wants to sanity-check the causal structure underlying a "
        "cohort before running identification."
    ),
)
async def causal_dag_infer(
    variables: list[str],
    data: list[list[float]],
    alpha: float = 0.05,
    max_observations: int = 5_000,
) -> dict[str, Any]:
    inp = DagInferInput(
        variables=variables,
        data=data,
        alpha=alpha,
        maxObservations=max_observations,
    )
    return await asyncio.to_thread(lambda: _envelope_to_dict(dag_infer(inp)))


@mcp.tool(
    name="explain-causal-evidence",
    description=(
        "Given a previously-produced CausalEvidence envelope, produce a "
        "plain-English paragraph (Claude translator; offline fallback "
        "when ANTHROPIC_API_KEY is unset). Use as the final step of any "
        "causal tool chain so the agent cites ready-made narrative."
    ),
)
async def explain_causal_evidence_tool(
    envelope: dict[str, Any],
    question: str,
    max_tokens: int = 400,
) -> dict[str, Any]:
    env_obj = CausalEvidenceEnvelope.model_validate(envelope)
    inp = ExplainInput(envelope=env_obj, question=question, max_tokens=max_tokens)
    result = await explain_causal_evidence(inp)
    return _envelope_to_dict(result)


def main() -> None:
    """Run the FastMCP server over SSE."""
    mcp.run(transport="sse")


if __name__ == "__main__":
    main()
