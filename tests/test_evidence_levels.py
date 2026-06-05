"""
Evidence-level classification tests — pure math on the sample-size ladders.

We test each engine's internal classifier so the calling agent gets
consistent signals across all 5 causal endpoints.
"""

from __future__ import annotations

from app.engines.dowhy_engine import _evidence_level_from_n as dowhy_level
from app.engines.econml_engine import _evidence_level_from_n as econml_level
from app.engines.pymc_engine import _evidence_level_bayesian as pymc_level


def test_dowhy_ladder():
    # Floor is set at 6 — see dowhy_engine._evidence_level_from_n.
    assert dowhy_level(0) == "insufficient"
    assert dowhy_level(5) == "insufficient"
    assert dowhy_level(6) == "preliminary"
    assert dowhy_level(9) == "preliminary"
    assert dowhy_level(10) == "preliminary"
    assert dowhy_level(29) == "preliminary"
    assert dowhy_level(30) == "moderate"
    assert dowhy_level(99) == "moderate"
    assert dowhy_level(100) == "strong"
    assert dowhy_level(10_000) == "strong"


def test_econml_ladder():
    # EconML needs more data than DoWhy because the forest needs variance.
    assert econml_level(19) == "insufficient"
    assert econml_level(20) == "preliminary"
    assert econml_level(49) == "preliminary"
    assert econml_level(50) == "moderate"
    assert econml_level(199) == "moderate"
    assert econml_level(200) == "strong"


def test_pymc_respects_diagnostics_over_sample_size():
    # r_hat > 1.05 downgrades to preliminary even with n=200.
    assert pymc_level(ess_bulk=1000, r_hat=1.06, n=200) == "preliminary"
    # ess_bulk < 200 same.
    assert pymc_level(ess_bulk=50, r_hat=1.0, n=200) == "preliminary"
    # Clean diagnostics + n>=50 → moderate.
    assert pymc_level(ess_bulk=1000, r_hat=1.01, n=60) == "moderate"
    # Clean diagnostics + n<50 → moderate too (clean is the gate).
    assert pymc_level(ess_bulk=1000, r_hat=1.0, n=20) == "moderate"
    # n<10 or non-finite → insufficient.
    assert pymc_level(ess_bulk=100, r_hat=1.0, n=5) == "insufficient"
    assert pymc_level(ess_bulk=float("nan"), r_hat=1.0, n=200) == "insufficient"
