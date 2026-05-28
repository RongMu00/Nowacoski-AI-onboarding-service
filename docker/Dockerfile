# Use Python 3.13 slim image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy everything needed for install
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

# Install CPU-only PyTorch first (avoids downloading ~2GB NVIDIA CUDA libs)
RUN uv pip install --system --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install the package (non-editable) + sentence-transformers
RUN uv pip install --system --no-cache-dir . \
    && uv pip install --system --no-cache-dir sentence-transformers
COPY .streamlit/ .streamlit/

# Expose Streamlit port
EXPOSE 8501

# Health check against Streamlit's built-in health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "src/enterprise_ai/ui.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
