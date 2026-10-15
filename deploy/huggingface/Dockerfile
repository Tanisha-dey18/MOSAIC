# MOSAIC backend for a Hugging Face Docker Space (free CPU tier: 2 vCPU, 16 GB RAM).
# This Space builds directly from the public GitHub repository; only this Dockerfile and
# README.md need to be in the Space. After pushing new code to GitHub, use
# "Settings -> Factory rebuild" in the Space to pick it up.
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 user

ARG MOSAIC_REPO=https://github.com/Tanisha-dey18/MOSAIC.git
ARG MOSAIC_REF=main
WORKDIR /app
RUN git clone --depth 1 --branch ${MOSAIC_REF} ${MOSAIC_REPO} /app

# CPU-only PyTorch (much smaller than the CUDA build), then the remaining pinned requirements.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
 && grep -v '^torch' backend/requirements.txt > /tmp/req.txt \
 && pip install --no-cache-dir -r /tmp/req.txt

# Free Spaces have no persistent disk: the database and processed data live in the container
# and MOSAIC_BOOTSTRAP rebuilds them (download -> ingest -> train) after every restart.
ENV MOSAIC_ENV=production \
    MOSAIC_BOOTSTRAP=1 \
    DATABASE_URL=sqlite:////data/storage/mosaic.db \
    STORAGE_DIR=/data/storage \
    DATASETS_DIR=/data/datasets \
    REPORTS_DIR=/app/reports \
    CORS_ORIGINS=https://mosaicappn.netlify.app \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data/storage /data/datasets && chown -R user:user /data /app
USER user
EXPOSE 7860
CMD ["sh", "-c", "cd backend && alembic upgrade head && cd /app && exec uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 7860 --workers 1"]
