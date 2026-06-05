"""
Min-n refusal tests — each engine has a gate below which it returns
``evidence_level=insufficient`` without invoking its heavyweight sampler.
These gates guard against unreliable convergence on tiny cohorts
(e.g. PyMC convergence on n<20). Regressions here mean the agent would
get misleading confidence.
"""

from __future__ import annotations

from app.engines.dowhy_engine import backdoor_check
from app.engines.econml_engine import hte_estimate
from app.engines.pymc_engine import bayesian_counterfactual
from app.schemas import (
    BackdoorCheckInput,
    BayesianCounterfactualInput,
    HteInput,
    ObservationSeries,
)


def test_backdoor_check_insufficient_on_empty_cohort():
    env = backdoor_check(
        BackdoorCheckInput(
            treatment="T",
            outcome="Y",
            observedConfounders=[],
            candidateConfounders=["covariate_b"],
            graphEdges=[("T", "Y")],
            cohort=[],
        )
    )
    assert env.evidence.evidence_level == "insufficient"
    assert env.evidence.observations == 0
    # Candidate confounders surface even when we refuse to compute.
    assert "covariate_b" in env.evidence.confounders


def test_backdoor_check_insufficient_on_malformed_rows():
    # Rows too short to align with the observed confounders.
    env = backdoor_check(
        BackdoorCheckInput(
            treatment="T",
            outcome="Y",
            observedConfounders=["weather"],
            candidateConfounders=["covariate_b"],
            graphEdges=[("T", "Y"), ("weather", "Y"), ("weather", "T")],
            cohort=[
                ObservationSeries(actorId="a1", values=[1.0]),  # missing outcome + weather
            ],
        )
    )
    assert env.evidence.evidence_level == "insufficient"


def test_bayesian_refuses_below_min_per_arm():
    env = bayesian_counterfactual(
        BayesianCounterfactualInput(
            treated=[1.0, 2.0, 3.0],  # n=3 < default 10
            untreated=[1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1],
        )
    )
    assert env.evidence.evidence_level == "insufficient"
    assert env.details["nTreated"] == 3
    assert env.details["minPerArm"] == 10


def test_hte_refuses_below_20_observations():
    env = hte_estimate(
        HteInput(
            treatment=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # n=10 < 20
            outcome=[0.0] * 10,
            features=[[0.0] for _ in range(10)],
            featureNames=["x"],
        )
    )
    assert env.evidence.evidence_level == "insufficient"
    assert env.evidence.observations == 10
