# Use Python 3.14 slim as the base image
FROM python:3.14-slim

# Install system dependencies for MPI and building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create data directory for mounting
RUN mkdir /data && chmod 777 /data

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project files
COPY . .

# Install the package and its dependencies using uv
RUN uv pip install --system --no-cache .

# Add a non-root user for security
RUN useradd -m pst
USER pst

# Volume for data persistence
VOLUME /data

# Default command
ENTRYPOINT ["stormtracker"]
CMD ["--help"]
