# Causal Evidence MCP — Python 3.12 + uv.
#
# Two-stage build:
#   1. builder: install deps with uv into a venv at /opt/venv
#   2. runtime: copy the venv + app source; drop privileges; expose 8787
#
# The runtime layer is deliberately trimmed. Notably we do NOT install
# git, gcc, or any build toolchain — all native extensions are resolved
# during the builder stage. That prevents the service from pulling in
# things it doesn't need to execute already-compiled code.

FROM python:3.12-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Build toolchain needed for dowhy, pymc, econml wheels that don't publish
# aarch64/x86_64 pre-builds for every Python minor.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv via its official static bootstrap.
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the manifest first so dependency install is cached independent
# of source edits.
COPY pyproject.toml README.md ./
COPY app ./app
# An empty lockfile is acceptable — uv syncs against pyproject on first
# build. For reproducibility in CI, commit the lock produced by the first
# clean build.
RUN uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && uv pip install -e .

# -----------

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# Runtime-only system deps: libgomp for OpenMP, ca-certs for Anthropic TLS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --shell /usr/sbin/nologin reasoning

COPY --from=builder --chown=reasoning:reasoning /opt/venv /opt/venv
COPY --from=builder --chown=reasoning:reasoning /app /app

WORKDIR /app
USER reasoning

EXPOSE 8787
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8787"]
