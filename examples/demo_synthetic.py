"""
Self-contained demo of the causal engines on purely synthetic data.

This script demonstrates the engines WITHOUT going through the HTTP
(FastAPI) or MCP (FastMCP) layer. It generates a synthetic two-arm
dataset with a known treatment effect, then calls two engine functions
directly:

    * app.engines.pymc_engine.bayesian_counterfactual
    * app.engines.dowhy_engine.backdoor_check

and pretty-prints the returned CausalEvidence envelope(s).

The data is entirely synthetic and self-generated — there is no external
input, no network access (the Bayesian + backdoor engines run fully
offline), and no domain context. Run it with:

    uv run python examples/demo_synthetic.py
"""

from __future__ import annotations

import json

import numpy as np

from app.engines.dowhy_engine import backdoor_check
from app.engines.pymc_engine import bayesian_counterfactual
from app.schemas import (
    BackdoorCheckInput,
    BayesianCounterfactualInput,
    CausalEvidenceEnvelope,
    ObservationSeries,
)

# A fixed seed so the demo is reproducible.
RNG = np.random.default_rng(7)

# Known ground-truth treatment effect baked into the synthetic data.
TRUE_EFFECT = 2.5
N_PER_ARM = 60


def make_two_arm_dataset() -> tuple[list[float], list[float]]:
    """Generate two arms (untreated / treated) with a known additive effect.

    The untreated arm is drawn from N(10, 1.5). The treated arm is the
    same baseline shifted up by TRUE_EFFECT. This is the simplest setup
    where the Bayesian counterfactual should recover a posterior mean
    close to TRUE_EFFECT.
    """
    untreated = RNG.normal(loc=10.0, scale=1.5, size=N_PER_ARM)
    treated = RNG.normal(loc=10.0 + TRUE_EFFECT, scale=1.5, size=N_PER_ARM)
    return treated.tolist(), untreated.tolist()


def make_backdoor_cohort() -> list[ObservationSeries]:
    """Build a cohort for backdoor-check.

    Each observation row is [treatment_indicator, outcome, covariate_a].
    A single observed confounder (``covariate_a``) influences both the
    treatment assignment and the outcome, so it should appear in the
    adjustment set DoWhy recommends.
    """
    cohort: list[ObservationSeries] = []
    n = N_PER_ARM * 2
    covariate = RNG.normal(loc=0.0, scale=1.0, size=n)
    # Treatment probability rises with the covariate -> confounding.
    treat_prob = 1.0 / (1.0 + np.exp(-covariate))
    treatment = (RNG.uniform(size=n) < treat_prob).astype(float)
    # Outcome depends on both treatment (the effect) and the covariate.
    outcome = 10.0 + TRUE_EFFECT * treatment + 1.2 * covariate + RNG.normal(scale=1.0, size=n)

    for i in range(n):
        cohort.append(
            ObservationSeries(
                actorId=f"unit-{i:03d}",
                values=[float(treatment[i]), float(outcome[i]), float(covariate[i])],
            )
        )
    return cohort


def _dump(envelope: CausalEvidenceEnvelope) -> str:
    return json.dumps(envelope.model_dump(by_alias=True), indent=2, default=str)


def run() -> None:
    print("=" * 70)
    print(f"Synthetic two-arm dataset: true effect = {TRUE_EFFECT}, "
          f"n = {N_PER_ARM} per arm")
    print("=" * 70)

    treated, untreated = make_two_arm_dataset()

    print("\n--- Bayesian counterfactual (PyMC NUTS) ---")
    bayes_envelope = bayesian_counterfactual(
        BayesianCounterfactualInput(treated=treated, untreated=untreated)
    )
    print(_dump(bayes_envelope))

    print("\n--- Backdoor check (DoWhy identify_effect) ---")
    cohort = make_backdoor_cohort()
    backdoor_envelope = backdoor_check(
        BackdoorCheckInput(
            treatment="treatment",
            outcome="outcome",
            observedConfounders=["covariate_a"],
            candidateConfounders=["covariate_a", "covariate_b"],
            graphEdges=[
                ("treatment", "outcome"),
                ("covariate_a", "treatment"),
                ("covariate_a", "outcome"),
            ],
            cohort=cohort,
        )
    )
    print(_dump(backdoor_envelope))


if __name__ == "__main__":
    run()
