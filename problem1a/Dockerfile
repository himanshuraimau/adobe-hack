# Multi-stage build for PDF Structure Extractor
# Platform: AMD64, CPU-only, no internet access required

FROM --platform=linux/amd64 python:3.12-slim as builder

# Set environment variables for CPU-only operation
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for PyMuPDF and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
# Force CPU-only PyTorch installation to reduce size and ensure CPU operation
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e .

# Production stage
FROM --platform=linux/amd64 python:3.12-slim

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV TORCH_DEVICE=cpu
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app

# Create working directory and set ownership
WORKDIR /app
RUN chown app:app /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=app:app *.py ./
COPY --chown=app:app __init__.py ./

# Copy the MobileBERT model (ensure it's under 200MB)
COPY --chown=app:app models/ ./models/

# Create input and output directories for volume mounting
RUN mkdir -p /app/input /app/output && \
    chown -R app:app /app/input /app/output

# Switch to non-root user
USER app

# Set up volume mount points
VOLUME ["/app/input", "/app/output"]

# Expose no ports (this is a batch processing application)

# Health check to verify the application can start
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app'); from main import main; print('OK')" || exit 1

# Default command - process all PDFs in input directory
CMD ["python", "main.py", "/app/input", "--output", "/app/output", "--timeout", "10"]

# Metadata
LABEL maintainer="PDF Structure Extractor"
LABEL description="Extracts structured outlines from PDF documents using MobileBERT"
LABEL version="1.0"
LABEL platform="linux/amd64"
LABEL cpu-only="true"