"""
PyMC-backed Bayesian counterfactual.

Model
-----
Two arms (treated / untreated) are modeled with Normal likelihoods whose
mean parameters have weakly-informative Normal priors centered on the
observed pooled mean. The treatment effect is the derived ``delta =
mu_treated - mu_untreated``, over which we report the posterior mean and
94% credible interval (ArviZ HDI).

Why NUTS vs a faster Gaussian conjugate formula: the conjugate answer is
closed-form only when priors + likelihood are both known Normal with
known variance. Real data has unknown variance, so NUTS over a Student-T
or a Half-Normal prior on sigma is the honest choice. With ``draws=2000``
and ``tune=1000`` convergence is typically reached in <10s for ≤200 obs
per arm on a laptop.

Min-n gate
----------
NUTS convergence is unreliable on small cohorts (n<20). Per-arm minimum
is configurable (default 10). Below that, we return
``evidence_level="insufficient"`` without running the sampler — NUTS
on n<3 produces non-finite CIs which the agent would mis-narrate.
"""

from __future__ import annotations

import numpy as np

from app.schemas import (
    BayesianCounterfactualInput,
    BayesianDetails,
    CausalEvidence,
    CausalEvidenceEnvelope,
)

_ENGINE_VERSION = "pymc-5.16"


def _evidence_level_bayesian(ess_bulk: float, r_hat: float, n: int) -> str:
    """Classify by posterior-quality metrics, not just sample size.

    For a Bayesian counterfactual, clean diagnostics (converged chains +
    large effective sample size) are the gate — not raw n. Once r_hat and
    ess_bulk are in range, the evidence caps at "moderate" regardless of
    how much data we collected. Escalation to "strong" requires an
    independent corroboration (e.g., matching frequentist CI), which this
    endpoint does not run.
    """
    if n < 10 or not np.isfinite(ess_bulk) or not np.isfinite(r_hat):
        return "insufficient"
    if r_hat > 1.05 or ess_bulk < 200:
        return "preliminary"
    return "moderate"


def bayesian_counterfactual(inp: BayesianCounterfactualInput) -> CausalEvidenceEnvelope:
    treated = np.asarray(inp.treated, dtype=float)
    untreated = np.asarray(inp.untreated, dtype=float)
    n_t = int(treated.size)
    n_u = int(untreated.size)
    n_total = n_t + n_u

    if n_t < inp.min_per_arm or n_u < inp.min_per_arm:
        return CausalEvidenceEnvelope(
            evidence=CausalEvidence(
                evidenceLevel="insufficient",
                observations=n_total,
                effectSize=0.0,
                pValue=None,
                confounders=[],
                methodology="bayesian_counterfactual_pymc",
                caveat=(
                    f"each arm requires ≥{inp.min_per_arm} observations "
                    f"(treated n={n_t}, untreated n={n_u}); sampler not invoked."
                ),
            ),
            details={
                "nTreated": n_t,
                "nUntreated": n_u,
                "minPerArm": inp.min_per_arm,
            },
            endpoint="/causal/bayesian-counterfactual",
            engineVersion=_ENGINE_VERSION,
        )

    # Lazy imports so tests that never hit this path don't pay the PyMC
    # startup cost.
    import arviz as az
    import pymc as pm

    pooled_mean = float(np.concatenate([treated, untreated]).mean())
    pooled_sd = float(np.concatenate([treated, untreated]).std(ddof=1))
    if pooled_sd == 0.0 or not np.isfinite(pooled_sd):
        pooled_sd = 1.0

    with pm.Model() as model:  # noqa: F841 - `model` is the active context
        mu_t = pm.Normal("mu_treated", mu=pooled_mean, sigma=pooled_sd * 3)
        mu_u = pm.Normal("mu_untreated", mu=pooled_mean, sigma=pooled_sd * 3)
        sigma_t = pm.HalfNormal("sigma_treated", sigma=pooled_sd * 2)
        sigma_u = pm.HalfNormal("sigma_untreated", sigma=pooled_sd * 2)

        pm.Normal("y_treated", mu=mu_t, sigma=sigma_t, observed=treated)
        pm.Normal("y_untreated", mu=mu_u, sigma=sigma_u, observed=untreated)

        delta = pm.Deterministic("delta", mu_t - mu_u)  # noqa: F841
        idata = pm.sample(
            draws=inp.draws,
            tune=inp.tune,
            chains=2,
            cores=1,
            progressbar=False,
            random_seed=20260421,
        )

    # Posterior summary via ArviZ. Column names for HDI bounds follow the
    # pattern `hdi_<lower>%` / `hdi_<upper>%` where the numbers drop trailing
    # zeros (ArviZ uses "3%" not "3.0%", but "2.5%" stays as-is).
    summary = az.summary(idata, var_names=["delta"], kind="all", hdi_prob=inp.credible_mass)
    row = summary.loc["delta"]
    posterior_mean = float(row["mean"])
    lower_pct = (1 - inp.credible_mass) / 2 * 100
    upper_pct = 100 - lower_pct

    def _hdi_col(pct: float) -> str:
        rendered = f"{pct:.1f}".rstrip("0").rstrip(".")
        return f"hdi_{rendered}%"

    hdi_lower = float(row[_hdi_col(lower_pct)])
    hdi_upper = float(row[_hdi_col(upper_pct)])
    ess_bulk = float(row["ess_bulk"])
    r_hat = float(row["r_hat"])

    # Cohen's d against the observed pooled sd.
    pooled_var = float(np.concatenate([treated, untreated]).var(ddof=1))
    effect_size = posterior_mean / np.sqrt(pooled_var) if pooled_var > 0 else 0.0

    caveat_bits: list[str] = []
    if r_hat > 1.05:
        caveat_bits.append(f"r_hat={r_hat:.3f} exceeds 1.05 — chains may not have converged")
    if ess_bulk < 200:
        caveat_bits.append(f"ess_bulk={ess_bulk:.0f} below 200 — posterior is thinly sampled")
    if inp.credible_mass != 0.94:
        caveat_bits.append(f"CI width set to {inp.credible_mass:.2f} (non-default)")
    if not caveat_bits:
        caveat_bits.append("posterior converged cleanly at the reported credible mass")

    evidence = CausalEvidence(
        evidenceLevel=_evidence_level_bayesian(ess_bulk, r_hat, n_total),
        observations=n_total,
        effectSize=float(effect_size),
        pValue=None,
        confounders=[],
        methodology="bayesian_counterfactual_pymc",
        caveat="; ".join(caveat_bits),
    )
    details = BayesianDetails(
        posteriorMean=posterior_mean,
        credibleInterval=(hdi_lower, hdi_upper),
        credibleMass=inp.credible_mass,
        essBulk=ess_bulk,
        rHat=r_hat,
    ).model_dump(by_alias=True)

    return CausalEvidenceEnvelope(
        evidence=evidence,
        details=details,
        endpoint="/causal/bayesian-counterfactual",
        engineVersion=_ENGINE_VERSION,
    )
