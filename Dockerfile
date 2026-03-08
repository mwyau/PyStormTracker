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

# Copy the project files
COPY . .

# Install the package and its dependencies
RUN pip install --no-cache-dir .

# Add a non-root user for security
RUN useradd -m pst
USER pst

# Volume for data persistence
VOLUME /data

# Default command
ENTRYPOINT ["stormtracker"]
CMD ["--help"]
