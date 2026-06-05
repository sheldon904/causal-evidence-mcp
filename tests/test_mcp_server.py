"""
Tests for the FastMCP server.

We don't stand up the SSE transport in unit tests — we call the tool
handlers directly and assert they dispatch to the engine functions with
the right argument shape. The HTTP surface is covered by the container
healthcheck.
"""

from __future__ import annotations

import pytest

from app.mcp_server import (
    causal_bayesian_counterfactual,
    causal_hte_estimate,
    mcp,
)


def test_mcp_server_name_is_reasoning():
    # FastMCP exposes `name` via the server attribute so Claude surfaces it
    # in tool-use traces. If this changes, the caller's tool-routing prompt
    # needs an update.
    assert mcp.name == "reasoning"


async def test_bayesian_tool_returns_camelcase_envelope_for_sufficient_input():
    result = await causal_bayesian_counterfactual(
        treated=[10, 12, 11, 13, 9, 14, 10, 11, 12, 13, 11, 10],
        untreated=[8, 7, 9, 8, 7, 8, 9, 8, 7, 9, 8, 7],
    )
    assert "evidence" in result
    assert "endpoint" in result
    assert result["endpoint"] == "/causal/bayesian-counterfactual"
    assert result["evidence"]["evidenceLevel"] in {"moderate", "strong", "preliminary"}
    ci = result["details"]["credibleInterval"]
    assert len(ci) == 2
    assert ci[0] < ci[1]


async def test_bayesian_tool_refuses_tiny_cohort():
    result = await causal_bayesian_counterfactual(
        treated=[1, 2, 3],  # n=3 < min_per_arm=10
        untreated=[1.1] * 12,
    )
    assert result["evidence"]["evidenceLevel"] == "insufficient"
    assert "sampler not invoked" in result["evidence"]["caveat"]


async def test_hte_tool_refuses_small_cohort():
    result = await causal_hte_estimate(
        treatment=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # n=10 < 20
        outcome=[0.0] * 10,
        features=[[0.0] for _ in range(10)],
        feature_names=["x"],
    )
    assert result["evidence"]["evidenceLevel"] == "insufficient"


def test_mcp_module_exports_every_plan_mandated_tool():
    # The service exposes 6 MCP tools. Fail loudly if the module drifts
    # out of sync.
    from app import mcp_server

    expected = {
        "causal_backdoor_check",
        "causal_sensitivity",
        "causal_bayesian_counterfactual",
        "causal_hte_estimate",
        "causal_dag_infer",
        "explain_causal_evidence_tool",
    }
    assert expected.issubset(set(dir(mcp_server)))
