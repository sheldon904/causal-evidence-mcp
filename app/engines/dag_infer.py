"""
DAG inference via the PC (Peter-Clark) constraint-based algorithm.

Why PC instead of NOTEARS: NOTEARS runtime on 10k+ observations is a
known performance concern. PC (from causal-learn) scales comfortably
into the 5k–10k range on typical cohort sizes AND has been Python-
mainlined longer than the NOTEARS reference implementations. The
function name stays neutral so a later upgrade to NOTEARS is a 1-file
change.

Output shape
------------
A list of ``DagEdge`` records with per-edge confidence (fraction of
bootstrap resamples that agreed on the edge) plus the usual
``CausalEvidence`` envelope. Direction is inferred from the CPDAG
returned by PC; bidirected edges (Markov-equivalent ambiguities) are
emitted as two directed edges with shared confidence.
"""

from __future__ import annotations

import numpy as np

from app.schemas import (
    CausalEvidence,
    CausalEvidenceEnvelope,
    DagEdge,
    DagInferInput,
)

_ENGINE_VERSION = "causal-learn-0.1.4"


def dag_infer(inp: DagInferInput) -> CausalEvidenceEnvelope:
    data = np.asarray(inp.data, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(inp.variables):
        raise ValueError(
            "data must be a 2-D array whose column count matches variables"
        )
    n_obs = data.shape[0]
    if n_obs < 20:
        return CausalEvidenceEnvelope(
            evidence=CausalEvidence(
                evidenceLevel="insufficient",
                observations=n_obs,
                effectSize=0.0,
                pValue=None,
                confounders=[],
                methodology="pc_algorithm_causal_learn",
                caveat=f"need ≥20 observations to run PC; got {n_obs}",
            ),
            endpoint="/causal/dag-infer",
            engineVersion=_ENGINE_VERSION,
        )

    # Subsample to cap runtime — beyond 10k rows PC gets slow; 5k is the
    # POC ceiling. Uniform random sampling preserves marginals.
    if n_obs > inp.max_observations:
        rng = np.random.default_rng(20260421)
        idx = rng.choice(n_obs, size=inp.max_observations, replace=False)
        data = data[idx]

    from causallearn.search.ConstraintBased.PC import pc

    cg = pc(data, alpha=inp.alpha, show_progress=False)
    edges = _extract_edges(cg, inp.variables)

    # Bootstrap-based confidence: resample 50 times, re-run PC, count edge
    # frequency. POC budget — configurable later.
    rng = np.random.default_rng(20260422)
    n_boot = min(50, max(10, int(data.shape[0] / 20)))
    edge_counts: dict[tuple[str, str], int] = {}
    for _ in range(n_boot):
        boot_idx = rng.integers(0, data.shape[0], size=data.shape[0])
        try:
            cg_b = pc(data[boot_idx], alpha=inp.alpha, show_progress=False)
        except Exception:
            continue
        for e in _extract_edges(cg_b, inp.variables):
            key = (e.source, e.target)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    scored: list[DagEdge] = []
    for e in edges:
        key = (e.source, e.target)
        count = edge_counts.get(key, 0)
        scored.append(
            DagEdge(
                source=e.source,
                target=e.target,
                confidence=count / n_boot if n_boot > 0 else 0.0,
            )
        )

    # Average bootstrap confidence as a very rough effect-size proxy.
    effect_size = (
        float(np.mean([e.confidence for e in scored])) if scored else 0.0
    )

    evidence = CausalEvidence(
        evidenceLevel=_evidence_level_from_n(n_obs),
        observations=n_obs,
        effectSize=effect_size,
        pValue=None,
        confounders=[],
        methodology="pc_algorithm_causal_learn",
        caveat=(
            "PC recovers a CPDAG — some edge directions are Markov-"
            "equivalent and reported as pairs. Bootstrap confidence "
            f"computed over {n_boot} resamples."
        ),
    )

    return CausalEvidenceEnvelope(
        evidence=evidence,
        details={
            "edges": [e.model_dump() for e in scored],
            "alpha": inp.alpha,
            "nObservations": n_obs,
            "subsampledTo": int(data.shape[0]),
            "bootstrapResamples": n_boot,
        },
        endpoint="/causal/dag-infer",
        engineVersion=_ENGINE_VERSION,
    )


def _extract_edges(cg: object, variables: list[str]) -> list[DagEdge]:
    """
    Pull directed edges out of a causal-learn CausalGraph. The library's
    API is matrix-based: ``cg.G.graph[i, j] == 1`` means i → j in their
    convention. Bidirected edges produce both (i,j) and (j,i).
    """
    G = getattr(cg, "G", cg)
    matrix = getattr(G, "graph", None)
    if matrix is None:
        return []
    arr = np.asarray(matrix)
    n = arr.shape[0]
    out: list[DagEdge] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # causal-learn directionality: graph[i,j] == 1 and graph[j,i] == -1
            # encodes i -> j. A full "i -> j and j -> i" undirected edge shows
            # up as both +1 (we split it into two directed edges).
            if arr[i, j] == 1:
                out.append(DagEdge(source=variables[i], target=variables[j], confidence=1.0))
    return out


def _evidence_level_from_n(n: int) -> str:
    if n < 20:
        return "insufficient"
    if n < 100:
        return "preliminary"
    if n < 1_000:
        return "moderate"
    return "strong"
