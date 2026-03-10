# --- Build Stage ---
FROM python:3.13-slim AS builder

# Prevent uv from creating a virtualenv that might be hard to move
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install build dependencies with cache mount for apt
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 1. Copy only dependency files for better layer caching.
COPY pyproject.toml uv.lock ./

# 2. Install third-party dependencies first (including grib extra).
# Use cache mount for uv to persist downloads and build artifacts.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --extra grib

# 3. Copy only necessary source files for the final installation step.
COPY src/ ./src/
COPY README.md ./

# 4. Final installation of the project package itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra grib


# --- Runtime Stage ---
FROM python:3.13-slim

WORKDIR /app

# Install ONLY the runtime shared libraries
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libeccodes0 \
    && rm -rf /var/lib/apt/lists/*

# Create data directory for mounting
RUN mkdir /data && chmod 777 /data

# Copy the environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Ensure the virtualenv is used by default
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Add a non-root user for security
RUN useradd -m pst
USER pst

# Volume for data persistence
VOLUME /data

# Default command
ENTRYPOINT ["stormtracker"]
CMD ["--help"]
