# Local builds use the latest uv image; CI passes a pinned version.
ARG UV_VERSION=latest
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# --- Build Stage ---
FROM python:3.14-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# ducc0 may need a C++17 compiler when no compatible wheel is available.
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --extra grib --extra netcdf4 --no-editable

COPY src/ ./src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra grib --extra netcdf4 --no-editable

# --- Runtime Stage ---
FROM python:3.14-slim

WORKDIR /app
RUN mkdir /data && chmod 777 /data

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

RUN useradd -m pst
USER pst

VOLUME /data
ENTRYPOINT ["stormtracker"]
CMD ["--help"]
