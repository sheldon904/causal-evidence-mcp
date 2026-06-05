"""Causal-inference engines for the causal-evidence service.

Each module implements one endpoint's math. Engines are pure functions
taking Pydantic inputs and returning ``CausalEvidenceEnvelope`` so the
FastAPI layer is a thin transport shim.
"""
