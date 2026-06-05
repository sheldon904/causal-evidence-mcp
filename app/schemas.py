"""
Pydantic models for the causal-evidence service.

Defines a stable evidence contract so any LLM agent can consume results
from every endpoint uniformly: each endpoint returns a
``CausalEvidenceEnvelope`` with the common ``evidence`` payload plus
endpoint-specific extras in ``details``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------- Shared types ----------


EvidenceLevel = Literal["strong", "moderate", "preliminary", "insufficient"]
"""How much weight to put on a result, from strong evidence down to insufficient."""


class CausalEvidence(BaseModel):
    """The common evidence contract returned by every endpoint.

    A stable, JSON-friendly shape so any LLM agent can consume causal
    results uniformly regardless of which engine produced them.
    """

    evidence_level: EvidenceLevel = Field(alias="evidenceLevel")
    observations: int
    effect_size: float = Field(alias="effectSize")
    p_value: float | None = Field(default=None, alias="pValue")
    confounders: list[str] = Field(default_factory=list)
    methodology: str
    caveat: str

    model_config = ConfigDict(populate_by_name=True, ser_by_alias=True)


class CausalEvidenceEnvelope(BaseModel):
    """Every endpoint returns this shape."""

    evidence: CausalEvidence
    details: dict[str, Any] = Field(default_factory=dict)
    emitted_at: datetime = Field(default_factory=datetime.utcnow, alias="emittedAt")
    endpoint: str
    engine_version: str = Field(alias="engineVersion")

    model_config = ConfigDict(populate_by_name=True, ser_by_alias=True)


# ---------- Endpoint inputs ----------


class ObservationSeries(BaseModel):
    """One series of observations for a single unit/operator."""

    actor_id: str = Field(alias="actorId")
    values: list[float]
    context: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class BackdoorCheckInput(BaseModel):
    """
    Ask: "given this cohort + a proposed confounder set, which of the
    confounders are *unblocked* by the observed graph?"
    """

    treatment: str
    outcome: str
    observed_confounders: list[str] = Field(alias="observedConfounders")
    candidate_confounders: list[str] = Field(alias="candidateConfounders")
    # A minimal adjacency list describing the causal DAG the caller believes in.
    # Each entry is a directed edge "parent -> child". Used by DoWhy to search
    # for backdoor paths.
    graph_edges: list[tuple[str, str]] = Field(alias="graphEdges")
    cohort: list[ObservationSeries] = Field(default_factory=list)
    # Hard minimum cohort size below which we refuse to run identification and
    # return evidence_level="insufficient". Mirrors BayesianCounterfactualInput.
    min_per_arm: int = Field(default=10, alias="minPerArm")

    model_config = ConfigDict(populate_by_name=True)


class SensitivityInput(BaseModel):
    """
    Ask: "how strong would a hidden confounder need to be to flip our
    conclusion?" Wraps DoWhy's `refute_estimate` with `add_unobserved_common_cause`.
    """

    treatment: str
    outcome: str
    graph_edges: list[tuple[str, str]] = Field(alias="graphEdges")
    cohort_treated: list[float] = Field(alias="cohortTreated")
    cohort_untreated: list[float] = Field(alias="cohortUntreated")
    # Strength of the hypothetical unobserved confounder as it pertains to T and Y.
    effect_strength_on_treatment: float = Field(
        default=0.1, alias="effectStrengthOnTreatment"
    )
    effect_strength_on_outcome: float = Field(
        default=0.1, alias="effectStrengthOnOutcome"
    )
    # Hard minimum per-arm cohort size below which we refuse to refute and
    # return evidence_level="insufficient". Mirrors BayesianCounterfactualInput.
    min_per_arm: int = Field(default=10, alias="minPerArm")

    model_config = ConfigDict(populate_by_name=True)


class BayesianCounterfactualInput(BaseModel):
    """
    Ask: "given these observed outcomes, what's the posterior credible
    interval on the treatment effect?"
    """

    treated: list[float]
    untreated: list[float]
    # Credible-interval width. 0.94 matches the PyMC / ArviZ default and the
    # service's Bayesian narrative ("94% CI").
    credible_mass: float = Field(default=0.94, alias="credibleMass")
    draws: int = 2000
    tune: int = 1000
    # Hard minimum cohort size (per arm) below which we refuse to sample and
    # return evidence_level="insufficient" with zero observations.
    min_per_arm: int = Field(default=10, alias="minPerArm")

    model_config = ConfigDict(populate_by_name=True)


class HteInput(BaseModel):
    """
    Ask: "does the treatment effect vary by context?" Uses EconML's
    CausalForestDML. `features` is the context matrix (one row per
    observation), `treatment` and `outcome` are parallel 1-D arrays.
    """

    treatment: list[int]  # 0/1 treatment indicator
    outcome: list[float]
    features: list[list[float]]
    feature_names: list[str] = Field(alias="featureNames")
    # Number of trees in the causal forest — default 100 is a reasonable POC.
    n_estimators: int = Field(default=100, alias="nEstimators")

    model_config = ConfigDict(populate_by_name=True)


class DagInferInput(BaseModel):
    """
    Ask: "given these observed variables, what's the likely DAG?"
    Uses the PC algorithm (constraint-based, discrete-friendly).
    """

    variables: list[str]
    # Matrix of observations: len(data) == n_observations,
    # len(data[i]) == len(variables).
    data: list[list[float]]
    alpha: float = 0.05
    # If observations exceed this count, subsample uniformly. NOTEARS on
    # >10k rows is a known performance concern; 5k is a safe POC ceiling.
    max_observations: int = Field(default=5_000, alias="maxObservations")

    model_config = ConfigDict(populate_by_name=True)


class ExplainInput(BaseModel):
    """
    Given a previously-produced CausalEvidenceEnvelope (any endpoint), call
    Anthropic to produce a plain-English paragraph. The calling agent uses
    this to narrate tool output without generating stats itself.
    """

    envelope: CausalEvidenceEnvelope
    question: str
    max_tokens: int = 400

    model_config = ConfigDict(populate_by_name=True)


# ---------- Endpoint outputs ----------


class BayesianDetails(BaseModel):
    """Extra payload attached to the Bayesian envelope."""

    posterior_mean: float = Field(alias="posteriorMean")
    credible_interval: tuple[float, float] = Field(alias="credibleInterval")
    credible_mass: float = Field(alias="credibleMass")
    ess_bulk: float = Field(alias="essBulk")
    r_hat: float = Field(alias="rHat")

    model_config = ConfigDict(populate_by_name=True, ser_by_alias=True)


class HteStratum(BaseModel):
    """One stratum of HTE output — a feature regime + its effect + CI."""

    feature: str
    threshold: float
    effect_below: float = Field(alias="effectBelow")
    effect_above: float = Field(alias="effectAbove")
    n_below: int = Field(alias="nBelow")
    n_above: int = Field(alias="nAbove")

    model_config = ConfigDict(populate_by_name=True, ser_by_alias=True)


class DagEdge(BaseModel):
    source: str
    target: str
    confidence: float


# ---------- Error response ----------


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
