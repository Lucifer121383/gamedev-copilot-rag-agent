FROM node:22-alpine AS frontend-build

WORKDIR /frontend
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm run build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_BACKEND=hashing \
    VECTOR_BACKEND=faiss \
    RERANKER_BACKEND=lightweight

WORKDIR /app

COPY requirements.txt requirements-production.txt ./
RUN pip install --no-cache-dir -r requirements-production.txt

COPY . .
COPY --from=frontend-build /web ./web

RUN mkdir -p /app/runtime-storage && \
    useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app

USER appuser
EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/healthz', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8010}"]
