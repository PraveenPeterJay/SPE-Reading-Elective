# ─────────────────────────────────────────────────────────────
# Python / ONNX Runtime inference lane
# Model: Pima Diabetes (tabular, 8 features)
# ─────────────────────────────────────────────────────────────

# 1. Base image – slim keeps attack surface small (fewer CVEs for Trivy)
FROM python:3.9-slim

# 2. Working directory
WORKDIR /app

# 3. System dependencies
#    libgl1 / libglib2.0-0 were only needed for Pillow/OpenCV (image model).
#    For a pure tabular ONNX model we need nothing extra — drop them entirely
#    to reduce the image size and CVE surface area (directly measurable in paper).
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# 4. Python dependencies – copy first for layer-cache efficiency
COPY python-app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5. Copy model artifacts (scaler + ONNX model)
#    Keeping them on a separate layer means code changes don't invalidate
#    the (larger) model layer cache.
COPY model_artifacts/model.onnx    /app/model_artifacts/model.onnx
COPY model_artifacts/scaler.json   /app/model_artifacts/scaler.json

# 6. Copy application code
COPY python-app/main.py .

# 7. Non-root user – reduces container breakout risk
#    This is a concrete security metric you can cite in the paper.
RUN useradd --no-create-home --shell /bin/false appuser
USER appuser

# 8. Port
EXPOSE 8000

# 9. Healthcheck – lets Docker (and your Ansible summary step) poll liveness
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 10. Entrypoint
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]