# ═══════════════════════════════════════════════════════════════
# STAGE 1 — BUILDER
# Ubuntu 24.04 gives us GLIBC 2.39, required by ORT 2.x
# ═══════════════════════════════════════════════════════════════
FROM ubuntu:24.04 AS builder
WORKDIR /usr/src/app

# System build tools — no image libs needed anymore
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# ── Dependency-cache layer ──────────────────────────────────────
# Copy manifests first; only re-runs cargo fetch when Cargo.toml changes
COPY rust-app/Cargo.toml rust-app/Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release 2>&1 | tail -5

# ── Real build ──────────────────────────────────────────────────
COPY rust-app/src ./src
RUN touch src/main.rs
RUN cargo build --release 2>&1 | tail -5

# ═══════════════════════════════════════════════════════════════
# STAGE 2 — RUNNER
# Minimal Ubuntu; same GLIBC version as builder
# No Rust toolchain, no build tools → smaller attack surface
# ═══════════════════════════════════════════════════════════════
FROM ubuntu:24.04
WORKDIR /app

# Only runtime libs; libssl for TLS, ca-certificates for HTTPS
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (CWE-250 mitigation — citable security metric)
RUN useradd --no-create-home --shell /bin/false appuser

# Model artifacts on their own layer (never changes → stays cached)
COPY model_artifacts/model.onnx   /app/model_artifacts/model.onnx
COPY model_artifacts/scaler.json  /app/model_artifacts/scaler.json

# Application binary
COPY --from=builder /usr/src/app/target/release/rust-app /app/rust-app
RUN chmod +x /app/rust-app

USER appuser

ENV LD_LIBRARY_PATH="/app:/usr/lib"
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD bash -c 'echo -e "GET /health HTTP/1.0\r\n" | nc localhost 8000 | grep -q "ok"' \
    || exit 1

CMD ["./rust-app"]