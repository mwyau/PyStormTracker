# --- Build Stage ---
FROM python:3.14-slim AS builder

ARG TARGETARCH

# Prevent uv from creating a virtualenv that might be hard to move
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install build dependencies
# Note: libfftw3-dev is only required for SHTns (amd64)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    make \
    libc6-dev \
    $( [ "$TARGETARCH" = "amd64" ] && echo "libfftw3-dev" ) \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 1. Copy only dependency files for better layer caching.
COPY pyproject.toml uv.lock ./

# 2. Install third-party dependencies first (including extras).
# Use cache mount for uv to persist downloads and build artifacts.
# uv automatically respects the platform markers for [shtns] in pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --extra grib --extra netcdf4 --extra shtns --no-editable

# 3. Copy only necessary source files for the final installation step.
COPY src/ ./src/
COPY README.md ./

# 4. Final installation of the project package itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra grib --extra netcdf4 --extra shtns --no-editable


# --- Runtime Stage ---
FROM python:3.14-slim

ARG TARGETARCH

WORKDIR /app

# Install runtime dependencies
# libfftw3-double3 is only required for SHTns (amd64)
RUN apt-get update && apt-get install -y --no-install-recommends \
    $( [ "$TARGETARCH" = "amd64" ] && echo "libfftw3-double3" ) \
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
