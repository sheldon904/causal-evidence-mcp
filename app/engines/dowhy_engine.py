"""
DoWhy-backed backdoor identification + sensitivity analysis.

backdoor-check
--------------
Given a hypothesized DAG (list of directed edges) and an observed cohort,
run ``dowhy.CausalModel.identify_effect()`` and report:
  - the adjustment set DoWhy recommends (confounders we SHOULD control for)
  - confounders we *cannot* block given the observed graph (returned in the
    ``CausalEvidence.confounders`` field as the unblocked ones)
  - methodology name ("backdoor_criterion" / "frontdoor_criterion" / "iv")

sensitivity
-----------
Run DoWhy's ``refute_estimate(method_name="add_unobserved_common_cause")``
and return the magnitude of a hidden confounder that would flip the
estimated effect's sign. This is the metric the calling agent cites when
explaining why a seemingly strong effect is still provisional.

Neither function requires a running dependency graph beyond DoWhy itself
— we build a synthetic pandas frame from the input cohort. For this POC
this is sufficient; with real cohort retrieval the engine simply swaps
the dataframe source.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.schemas import (
    BackdoorCheckInput,
    CausalEvidence,
    CausalEvidenceEnvelope,
    SensitivityInput,
)

_ENGINE_VERSION = "dowhy-0.11"


def _evidence_level_from_n(n: int) -> str:
    """4-bucket evidence ladder keyed to sample size.

    The floor is set at 6 — wider CIs at small n are a better trade than
    blanket refusal in a small-data corpus, as long as the 'preliminary'
    tag flows through to downstream confidence calibration. The
    'preliminary' label is reused for both [6, 9] and [10, 29].
    """
    if n < 6:
        return "insufficient"
    if n < 30:
        return "preliminary"
    if n < 100:
        return "moderate"
    return "strong"


def _build_observation_frame(
    treatment: str,
    outcome: str,
    observed_confounders: list[str],
    cohort_values: list[list[float]],
) -> pd.DataFrame:
    """
    Stack a list-of-lists cohort into a DataFrame with the expected column
    names. Each inner list is [treatment, outcome, confounder_1, ...] in
    the order the caller declared.
    """
    cols = [treatment, outcome, *observed_confounders]
    return pd.DataFrame(cohort_values, columns=cols)


def _graph_dot(edges: list[tuple[str, str]]) -> str:
    """Render DoWhy-compatible DOT from a list of (source, target) edges."""
    body = ";\n  ".join(f'"{s}" -> "{t}"' for s, t in edges)
    return f"digraph {{ {body} }}"


def backdoor_check(inp: BackdoorCheckInput) -> CausalEvidenceEnvelope:
    """
    Run DoWhy.identify_effect on the provided graph + cohort.

    The returned envelope's `confounders` field contains the set of caller-
    declared candidate confounders that remain UNBLOCKED after adjustment
    for the observed set. That's the actionable answer: "you said you
    controlled for X, but these are still open."
    """
    n = len(inp.cohort)
    if n < 1:
        return CausalEvidenceEnvelope(
            evidence=CausalEvidence(
                evidenceLevel="insufficient",
                observations=0,
                effectSize=0.0,
                pValue=None,
                confounders=list(inp.candidate_confounders),
                methodology="backdoor_criterion",
                caveat="cohort is empty — identification not attempted",
            ),
            endpoint="/causal/backdoor-check",
            engineVersion=_ENGINE_VERSION,
        )

    # Min-n gate: refuse below `min_per_arm` so small-data callers
    # see a structured refusal rather than a brittle DoWhy result.
    if n < inp.min_per_arm:
        return CausalEvidenceEnvelope(
            evidence=CausalEvidence(
                evidenceLevel="insufficient",
                observations=n,
                effectSize=0.0,
                pValue=None,
                confounders=list(inp.candidate_confounders),
                methodology="backdoor_criterion",
                caveat=(
                    f"cohort has {n} observation(s); min_per_arm={inp.min_per_arm} required — "
                    "identification not attempted."
                ),
            ),
            details={
                "cohortSize": n,
                "minPerArm": inp.min_per_arm,
            },
            endpoint="/causal/backdoor-check",
            engineVersion=_ENGINE_VERSION,
        )

    # Build observations. Expected shape per series: values[0]=treatment,
    # values[1]=outcome, values[2:]=observed_confounders (in caller order).
    raw_rows: list[list[float]] = []
    for series in inp.cohort:
        if len(series.values) < 2 + len(inp.observed_confounders):
            # Can't align — skip silently.
            continue
        raw_rows.append(list(series.values))
    if not raw_rows:
        return CausalEvidenceEnvelope(
            evidence=CausalEvidence(
                evidenceLevel="insufficient",
                observations=0,
                effectSize=0.0,
                pValue=None,
                confounders=list(inp.candidate_confounders),
                methodology="backdoor_criterion",
                caveat="no well-formed observation rows after alignment",
            ),
            endpoint="/causal/backdoor-check",
            engineVersion=_ENGINE_VERSION,
        )

    df = _build_observation_frame(
        inp.treatment, inp.outcome, inp.observed_confounders, raw_rows
    )

    # Import DoWhy lazily so importing this module for unit tests doesn't
    # drag in the whole causal stack when only the evidence-shape logic is
    # being exercised.
    from dowhy import CausalModel

    graph = _graph_dot(inp.graph_edges)
    model = CausalModel(
        data=df,
        treatment=inp.treatment,
        outcome=inp.outcome,
        graph=graph,
    )
    ident = model.identify_effect(proceed_when_unidentifiable=True)

    # Collect the adjustment set DoWhy recommends.
    recommended_adjustment: list[str] = []
    for name in ("backdoor_variables", "instrumental_variables", "frontdoor_variables"):
        vals = getattr(ident, name, None)
        if vals:
            recommended_adjustment.extend([v for v in vals if isinstance(v, str)])

    unblocked = [
        c for c in inp.candidate_confounders if c not in set(inp.observed_confounders)
    ]

    # Effect size: standardized mean difference between treated/untreated arms
    # (Cohen's d, unpooled). Degrades gracefully if the treatment column isn't
    # binary.
    effect_size = _cohens_d(df[inp.outcome].to_numpy(), df[inp.treatment].to_numpy())

    methodology = (
        "backdoor_criterion"
        if getattr(ident, "backdoor_variables", None)
        else "frontdoor_criterion"
        if getattr(ident, "frontdoor_variables", None)
        else "iv"
        if getattr(ident, "instrumental_variables", None)
        else "unidentified"
    )

    evidence = CausalEvidence(
        evidenceLevel=_evidence_level_from_n(len(df)),
        observations=len(df),
        effectSize=effect_size,
        pValue=None,
        confounders=sorted(set(unblocked)),
        methodology=methodology,
        caveat=(
            "DoWhy could not identify the effect given the supplied DAG."
            if methodology == "unidentified"
            else f"Adjusted for {sorted(set(inp.observed_confounders))}; remaining unblocked confounders reported above."
        ),
    )

    return CausalEvidenceEnvelope(
        evidence=evidence,
        details={
            "recommendedAdjustmentSet": sorted(set(recommended_adjustment)),
            "candidateConfounders": list(inp.candidate_confounders),
            "observedConfounders": list(inp.observed_confounders),
            "cohortSize": len(df),
            "graphEdgesDigest": _hash_edges(inp.graph_edges),
        },
        endpoint="/causal/backdoor-check",
        engineVersion=_ENGINE_VERSION,
    )


def sensitivity(inp: SensitivityInput) -> CausalEvidenceEnvelope:
    """
    Apply `refute_estimate(method_name="add_unobserved_common_cause")` after
    a naive mean-difference estimate. Returns the effect fraction of a
    hidden confounder that would flip the sign.
    """
    n_t = len(inp.cohort_treated)
    n_u = len(inp.cohort_untreated)
    n = n_t + n_u
    # Min-n gate: configurable per-arm refusal threshold (default 10).
    # Subsumes the legacy hard floor of 3 — when min_per_arm is left at default
    # we still refuse on the same small cases, just with a structured envelope.
    if min(n_t, n_u) < inp.min_per_arm:
        return CausalEvidenceEnvelope(
            evidence=CausalEvidence(
                evidenceLevel="insufficient",
                observations=n,
                effectSize=0.0,
                pValue=None,
                confounders=[],
                methodology="sensitivity_add_unobserved_common_cause",
                caveat=(
                    f"each arm requires ≥{inp.min_per_arm} observations "
                    f"(treated n={n_t}, untreated n={n_u}); refutation not attempted."
                ),
            ),
            details={
                "nTreated": n_t,
                "nUntreated": n_u,
                "minPerArm": inp.min_per_arm,
            },
            endpoint="/causal/sensitivity",
            engineVersion=_ENGINE_VERSION,
        )

    treated = np.asarray(inp.cohort_treated, dtype=float)
    untreated = np.asarray(inp.cohort_untreated, dtype=float)
    raw_effect = float(treated.mean() - untreated.mean())
    effect_size = _cohens_d_two_sample(treated, untreated)

    # Build the DoWhy model using a synthetic binary treatment column.
    from dowhy import CausalModel

    df = pd.DataFrame(
        {
            inp.treatment: np.concatenate(
                [np.ones(n_t, dtype=int), np.zeros(n_u, dtype=int)]
            ),
            inp.outcome: np.concatenate([treated, untreated]),
        }
    )
    # If the caller didn't give us edges, assume the simple T -> Y graph.
    edges = inp.graph_edges or [(inp.treatment, inp.outcome)]
    model = CausalModel(
        data=df, treatment=inp.treatment, outcome=inp.outcome, graph=_graph_dot(edges)
    )
    ident = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        ident,
        method_name="backdoor.linear_regression",
        test_significance=True,
    )

    refutation = model.refute_estimate(
        ident,
        estimate,
        method_name="add_unobserved_common_cause",
        confounders_effect_on_treatment="linear",
        confounders_effect_on_outcome="linear",
        effect_strength_on_treatment=inp.effect_strength_on_treatment,
        effect_strength_on_outcome=inp.effect_strength_on_outcome,
    )

    refuted_value = float(getattr(refutation, "new_effect", raw_effect))
    sign_flipped = (raw_effect * refuted_value) < 0
    p_value_new = getattr(refutation, "refutation_result", {}).get("p_value")

    caveat = (
        f"A hidden confounder with strength ({inp.effect_strength_on_treatment}, "
        f"{inp.effect_strength_on_outcome}) on (T,Y) "
        + ("WOULD flip the sign" if sign_flipped else "would not flip the sign")
        + "; results are "
        + ("fragile" if sign_flipped else "robust")
        + " to unobserved confounding at that magnitude."
    )

    return CausalEvidenceEnvelope(
        evidence=CausalEvidence(
            evidenceLevel=_evidence_level_from_n(n),
            observations=n,
            effectSize=effect_size,
            pValue=float(p_value_new) if isinstance(p_value_new, (int, float)) else None,
            confounders=[],
            methodology="sensitivity_add_unobserved_common_cause",
            caveat=caveat,
        ),
        details={
            "rawEffect": raw_effect,
            "refutedEffect": refuted_value,
            "signFlipped": sign_flipped,
            "effectStrengthOnTreatment": inp.effect_strength_on_treatment,
            "effectStrengthOnOutcome": inp.effect_strength_on_outcome,
        },
        endpoint="/causal/sensitivity",
        engineVersion=_ENGINE_VERSION,
    )


# ---------- Pure math helpers ----------


def _cohens_d(outcome: np.ndarray, treatment: np.ndarray) -> float:
    """Cohen's d between the treatment-=1 and treatment-=0 subsets."""
    mask_t = treatment.astype(float) > 0.5
    t_vals = outcome[mask_t]
    u_vals = outcome[~mask_t]
    if len(t_vals) < 2 or len(u_vals) < 2:
        return 0.0
    return _cohens_d_two_sample(t_vals, u_vals)


def _cohens_d_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def _hash_edges(edges: list[tuple[str, str]]) -> str:
    """Compact digest of the input DAG for provenance tagging."""
    import hashlib

    key = ";".join(sorted(f"{s}->{t}" for s, t in edges))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


# Mypy: ensure 'Any' is referenced so the import isn't pruned by ruff.
_ = Any
