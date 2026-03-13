# ── WASM inference lane ───────────────────────────────────────────────────────
# Compiles a Python fastapi server to WASM via wasm32-wasi using wasmtime.
# We use a lightweight Node + wasmtime-based WASM host instead of CPython
# to demonstrate the pure WASM sandbox execution model.
# ─────────────────────────────────────────────────────────────────────────────

FROM node:20-slim AS base

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install wasmtime CLI for running WASM modules
RUN curl -L https://github.com/bytecodealliance/wasmtime/releases/download/v18.0.3/\
wasmtime-v18.0.3-x86_64-linux.tar.xz \
    | tar -xJ -C /usr/local/bin --strip-components=1

WORKDIR /app

# Install wasm3 via npm as a JS-accessible WASM runtime (for HTTP serving)
COPY lanes/wasm/package.json ./package.json
RUN npm install --production

COPY model_artifacts/model.onnx  ./model.onnx
COPY model_artifacts/scaler.json ./scaler.json
COPY lanes/wasm/server.js        ./server.js
COPY lanes/wasm/infer.js         ./infer.js

# WASM note: in this lane, inference runs inside a WASM sandbox via wasmtime
# with WASI enabled. The Node host spawns wasmtime as a subprocess per-batch,
# demonstrating the WASM execution boundary and its security implications.
# Production WASM inference would use wasm-nn proposal; this models the sandbox.

RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8502
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
    CMD node -e "require('http').get('http://localhost:8502/health', r => process.exit(r.statusCode===200?0:1))"

CMD ["node", "server.js"]