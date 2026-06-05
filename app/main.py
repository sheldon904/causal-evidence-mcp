"""
FastAPI application — routes the causal-inference endpoints.

Each route is a thin shim that hands off to one engine module. No
business logic lives here; errors from the engines propagate with
explicit status codes so the calling agent can differentiate "bad input"
from "compute failure".
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Callable, TypeVar

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

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
    SensitivityInput,
)

# CPU-bound engines (PyMC, EconML, NOTEARS/PC) run on a worker pool so the
# event loop stays responsive to concurrent callers. The pool sizes itself
# by CPU count at startup.
_EXECUTOR: ThreadPoolExecutor | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _EXECUTOR
    _EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reasoning-cpu")
    try:
        yield
    finally:
        assert _EXECUTOR is not None
        _EXECUTOR.shutdown(wait=True)
        _EXECUTOR = None


app = FastAPI(
    title="Causal Evidence MCP",
    version="0.1.0",
    description=(
        "DoWhy/PyMC/EconML causal-inference endpoints. "
        "Advisory — produces evidence to inform decisions, not to replace them."
    ),
    lifespan=lifespan,
)


T = TypeVar("T")


def _executor() -> ThreadPoolExecutor:
    assert _EXECUTOR is not None, "ThreadPoolExecutor not initialized — lifespan missed?"
    return _EXECUTOR


async def _run_cpu(fn: Callable[[], T]) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor(), fn)


# --------- Health / introspection ---------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "engines": "dowhy,pymc,econml,causal-learn"}


@app.get("/")
async def root() -> dict[str, list[str]]:
    return {
        "endpoints": [
            "/causal/backdoor-check",
            "/causal/sensitivity",
            "/causal/bayesian-counterfactual",
            "/causal/hte-estimate",
            "/causal/dag-infer",
            "/explain/causal-evidence",
        ]
    }


# --------- Causal endpoints ---------


@app.post("/causal/backdoor-check", response_model=CausalEvidenceEnvelope)
async def backdoor_check_route(body: BackdoorCheckInput) -> CausalEvidenceEnvelope:
    try:
        return await _run_cpu(lambda: backdoor_check(body))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/causal/sensitivity", response_model=CausalEvidenceEnvelope)
async def sensitivity_route(body: SensitivityInput) -> CausalEvidenceEnvelope:
    try:
        return await _run_cpu(lambda: sensitivity(body))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/causal/bayesian-counterfactual", response_model=CausalEvidenceEnvelope)
async def bayesian_route(body: BayesianCounterfactualInput) -> CausalEvidenceEnvelope:
    try:
        return await _run_cpu(lambda: bayesian_counterfactual(body))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/causal/hte-estimate", response_model=CausalEvidenceEnvelope)
async def hte_route(body: HteInput) -> CausalEvidenceEnvelope:
    try:
        return await _run_cpu(lambda: hte_estimate(body))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/causal/dag-infer", response_model=CausalEvidenceEnvelope)
async def dag_infer_route(body: DagInferInput) -> CausalEvidenceEnvelope:
    try:
        return await _run_cpu(lambda: dag_infer(body))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --------- Explain ---------


@app.post("/explain/causal-evidence", response_model=CausalEvidenceEnvelope)
async def explain_route(body: ExplainInput) -> CausalEvidenceEnvelope:
    # Explain is IO-bound (Anthropic API) — don't use the CPU pool.
    try:
        return await explain_causal_evidence(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — surface provider errors to callers
        raise HTTPException(status_code=502, detail=f"explain upstream: {exc}") from exc
