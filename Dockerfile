# Base image with Python 3.10 (compatible with mediapipe 0.10.x)
FROM python:3.10.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    ENABLE_PDF_REPORTS=false

# System deps for OpenCV, fonts (matplotlib), wkhtmltopdf optional block commented out
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# (Optional) Enable PDF generation by uncommenting these lines and setting ENABLE_PDF_REPORTS=true
# RUN apt-get update && apt-get install -y --no-install-recommends wkhtmltopdf && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (leverage layer caching)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY . .

# Expose default Streamlit port (Render will still inject PORT env var)
EXPOSE 8501

# Healthcheck (simple: ensure process responds)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start Streamlit binding to provided PORT
CMD ["bash", "-c", "streamlit run streamlit_app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
