# Causal Evidence MCP

A small service that exposes six causal-inference tools over **two** surfaces:

- a **FastAPI HTTP** API on port **8787**, and
- an **MCP / SSE** surface (FastMCP) on port **4001**

so that LLM agents (including Anthropic remote-MCP compatible clients) can call
rigorous causal methods as tools. Every tool returns the same uniform
`CausalEvidence` envelope, so an agent can consume results from any endpoint the
same way.

## Tools

| Tool | Method | Returns |
|---|---|---|
| `causal-backdoor-check` | DoWhy `identify_effect` | adjustment set + candidate confounders still unblocked |
| `causal-sensitivity` | DoWhy `refute_estimate` with `add_unobserved_common_cause` | hidden-confounder strength that would flip the sign |
| `causal-bayesian-counterfactual` | PyMC NUTS sampler | posterior mean + 94% credible interval + ESS + R-hat |
| `causal-hte-estimate` | EconML `CausalForestDML` | per-feature median-split strata (effect below / above) |
| `causal-dag-infer` | causal-learn PC algorithm | edge list with bootstrap confidence scores |
| `explain-causal-evidence` | optional Anthropic narrative | plain-English paragraph (offline fallback when no API key) |

## Why

Causal inference is hard to do correctly inside an LLM. This service exposes
well-established causal methods (DoWhy, PyMC, EconML, causal-learn) as tools the
agent can call instead of trying to reason about confounding and counterfactuals
from raw text.

Every result returns a uniform `CausalEvidence` envelope carrying an
`evidence_level` (`strong` / `moderate` / `preliminary` / `insufficient`) and a
`caveat`. The evidence level is gated on sample size and posterior diagnostics,
and each engine refuses to run below a minimum cohort size — so the agent gets a
structured "insufficient" answer rather than a brittle or overconfident number.

## Local dev

```bash
# Install dependencies
uv sync

# HTTP surface (FastAPI) on port 8787
uv run uvicorn app.main:app --port 8787

# MCP / SSE surface (FastMCP) on port 4001
uv run python -m app.mcp_server
```

The only endpoint that makes a network call is `explain-causal-evidence`, and
only when `ANTHROPIC_API_KEY` is set. Every other endpoint runs fully offline.

## Docker

```bash
docker build -t causal-evidence-mcp .

# HTTP surface (default CMD exposes 8787)
docker run --rm -p 8787:8787 causal-evidence-mcp
```

## Demo

A self-contained example generates a synthetic two-arm dataset with a known
effect and calls the engines directly — no HTTP or MCP layer required:

```bash
uv run python examples/demo_synthetic.py
```

## Status

Advisory / POC-grade. The engines implement standard methods and gate their
output on sample-size and convergence diagnostics, but results are intended to
inform decisions, not replace expert judgment. Treat the `caveat` and
`evidence_level` fields as first-class.
