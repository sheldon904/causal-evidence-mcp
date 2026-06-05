"""
EconML-backed heterogeneous treatment effect estimator.

Wraps ``econml.dml.CausalForestDML`` to answer: "does this decision's
effect vary across contextual regimes (e.g. group A vs group B)?"

We use a default-parameter CausalForestDML because this project's
cohorts are typically in the 20–200 range where hyperparameter tuning is
noise. For bigger cohorts the caller can override ``n_estimators``.

Output shape
------------
A list of "strata" — one per feature — giving the treatment effect
estimated BELOW and ABOVE the feature's median value. This lets the
agent say things like "for units below the threshold, the effect is
+12%; above it drops to +2%". The full per-sample treatment effect
vector is also returned for callers that want to do their own slicing.
"""

from __future__ import annotations

import numpy as np

from app.schemas import (
    CausalEvidence,
    CausalEvidenceEnvelope,
    HteInput,
    HteStratum,
)

_ENGINE_VERSION = "econml-0.15"


def hte_estimate(inp: HteInput) -> CausalEvidenceEnvelope:
    T = np.asarray(inp.treatment, dtype=float)
    Y = np.asarray(inp.outcome, dtype=float)
    X = np.asarray(inp.features, dtype=float)
    n = X.shape[0]

    if n < 20 or Y.size != n or T.size != n:
        return CausalEvidenceEnvelope(
            evidence=CausalEvidence(
                evidenceLevel="insufficient",
                observations=n,
                effectSize=0.0,
                pValue=None,
                confounders=[],
                methodology="hte_causal_forest_dml",
                caveat=(
                    "HTE estimation skipped: at least 20 aligned observations "
                    f"required across treatment/outcome/features (got n={n})."
                ),
            ),
            endpoint="/causal/hte-estimate",
            engineVersion=_ENGINE_VERSION,
        )

    if X.ndim != 2 or X.shape[1] != len(inp.feature_names):
        raise ValueError(
            "features must be a 2-D array whose column count matches feature_names"
        )

    # Lazy import — EconML has heavy transitive deps.
    from econml.dml import CausalForestDML

    forest = CausalForestDML(
        n_estimators=inp.n_estimators,
        min_samples_leaf=5,
        random_state=20260421,
        discrete_treatment=True,
    )
    forest.fit(Y=Y, T=T, X=X)
    tau = forest.effect(X)  # per-sample treatment effect

    strata = _per_feature_median_split(X, tau, inp.feature_names)
    # Overall effect size: Cohen's d of tau around zero.
    effect_size = float(tau.mean() / tau.std(ddof=1)) if tau.std(ddof=1) > 0 else 0.0

    evidence = CausalEvidence(
        evidenceLevel=_evidence_level_from_n(n),
        observations=n,
        effectSize=effect_size,
        pValue=None,
        confounders=[],
        methodology="hte_causal_forest_dml",
        caveat=(
            "Per-sample treatment effects computed via CausalForestDML; "
            "strata are median-split summaries — ask for a richer slice "
            "via `/causal/hte-estimate` with custom thresholds if needed."
        ),
    )

    return CausalEvidenceEnvelope(
        evidence=evidence,
        details={
            "strata": [s.model_dump(by_alias=True) for s in strata],
            "tauMean": float(tau.mean()),
            "tauStd": float(tau.std(ddof=1)) if tau.size > 1 else 0.0,
            "tauMin": float(tau.min()),
            "tauMax": float(tau.max()),
            "nSamples": n,
            "nFeatures": X.shape[1],
        },
        endpoint="/causal/hte-estimate",
        engineVersion=_ENGINE_VERSION,
    )


def _per_feature_median_split(
    X: np.ndarray, tau: np.ndarray, feature_names: list[str]
) -> list[HteStratum]:
    out: list[HteStratum] = []
    for i, name in enumerate(feature_names):
        col = X[:, i]
        threshold = float(np.median(col))
        below_mask = col < threshold
        above_mask = ~below_mask
        if below_mask.sum() == 0 or above_mask.sum() == 0:
            continue
        out.append(
            HteStratum(
                feature=name,
                threshold=threshold,
                effectBelow=float(tau[below_mask].mean()),
                effectAbove=float(tau[above_mask].mean()),
                nBelow=int(below_mask.sum()),
                nAbove=int(above_mask.sum()),
            )
        )
    return out


def _evidence_level_from_n(n: int) -> str:
    if n < 20:
        return "insufficient"
    if n < 50:
        return "preliminary"
    if n < 200:
        return "moderate"
    return "strong"
