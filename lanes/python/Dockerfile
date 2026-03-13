# ── Python inference lane ─────────────────────────────────────────────────────
# Uses python:3.11-slim (not full) to minimise attack surface.
# No torch at runtime — only onnxruntime for inference.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="SPE Python Inference Lane"
LABEL org.opencontainers.image.description="ONNX Runtime inference via FastAPI, Python 3.11-slim"

# Pinned versions for SBOM reproducibility
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    onnxruntime==1.17.3 \
    numpy==1.26.4 \
    python-json-logger==2.0.7

WORKDIR /app
COPY model_artifacts/model.onnx  ./model.onnx
COPY model_artifacts/scaler.json ./scaler.json
COPY lanes/python/server.py      ./server.py

# Drop root — run as unprivileged user
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8500
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8500/health')"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8500", \
     "--workers", "2", "--log-level", "warning"]