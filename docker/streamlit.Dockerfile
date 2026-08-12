# ---- builder ----
FROM python:3.13-slim@sha256:84a57da03fbb4a77e8769a3d5b692ee8f1d43a319eb6eee7c2e0b39caf406bb8 AS builder
WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-streamlit.txt .
RUN pip install --no-cache-dir -r requirements-streamlit.txt

# ---- runtime ----
FROM python:3.13-slim@sha256:84a57da03fbb4a77e8769a3d5b692ee8f1d43a319eb6eee7c2e0b39caf406bb8
WORKDIR /app

LABEL org.opencontainers.image.title="misinformation-agent-ui" \
      org.opencontainers.image.description="Streamlit chat UI for the misinformation-agent API" \
      org.opencontainers.image.source="https://github.com/etahir2005/misinformation-agent"

RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m appuser

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --chown=appuser:appuser app.py .

ENV PATH="/opt/venv/bin:$PATH" \
    HOME=/home/appuser \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
