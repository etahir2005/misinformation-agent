# ---- builder ----
FROM python:3.13-slim@sha256:84a57da03fbb4a77e8769a3d5b692ee8f1d43a319eb6eee7c2e0b39caf406bb8 AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-api.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-api.txt

# ---- runtime ----
FROM python:3.13-slim@sha256:84a57da03fbb4a77e8769a3d5b692ee8f1d43a319eb6eee7c2e0b39caf406bb8
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="misinformation-agent-api" \
      org.opencontainers.image.description="LangGraph fact-checking agent HTTP API" \
      org.opencontainers.image.source="https://github.com/etahir2005/misinformation-agent"

RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m appuser

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --chown=appuser:appuser agent/ agent/
COPY --chown=appuser:appuser graph.py .
COPY --chown=appuser:appuser main.py .

ENV PATH="/opt/venv/bin:$PATH" \
    HOME=/home/appuser \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1

# Pre-download both sentence-transformers models used by
# agent/tools/vector_lookup_tool.py (the embedding model and the
# CrossEncoder reranker) at build time, baked into the image — so the
# running container never needs live internet access to Hugging Face, and
# the first real request isn't the one paying a slow cold-download cost.
# Only the *default* RERANKER_MODEL_NAME is pre-baked; an operator who
# overrides that env var at runtime falls back to a normal on-demand
# download for that one, same as if this step didn't exist at all.
RUN mkdir -p $HF_HOME && chown -R appuser:appuser $HF_HOME
USER appuser
RUN HF_HUB_OFFLINE=0 python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-base-en-v1.5'); \
CrossEncoder('BAAI/bge-reranker-base')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
