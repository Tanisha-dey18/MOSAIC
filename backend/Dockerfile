# MOSAIC backend (FastAPI + ML pipeline). Build from the repository root:
#   docker build -f backend/Dockerfile -t mosaic-backend .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    STORAGE_DIR=/data/storage \
    DATASETS_DIR=/data/datasets \
    REPORTS_DIR=/data/reports

WORKDIR /app

# CPU-only PyTorch wheel first (much smaller than the default CUDA build), then the rest.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
 && grep -v '^torch' backend/requirements.txt > /tmp/req.txt \
 && pip install -r /tmp/req.txt

COPY ml ml
COPY backend backend
COPY scripts scripts

RUN useradd --create-home --uid 10001 mosaic \
 && mkdir -p /data/storage /data/datasets /data/reports \
 && chown -R mosaic:mosaic /data /app
USER mosaic

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

# Apply database migrations, then start the API.
CMD ["sh", "-c", "cd backend && alembic upgrade head && cd /app && exec uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
