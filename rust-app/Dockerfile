# ═══════════════════════════════════════════════════════════════
# STAGE 1 — BUILDER
# ═══════════════════════════════════════════════════════════════
FROM ubuntu:24.04 AS builder
WORKDIR /usr/src/app

# Install system deps + Rust + build app in ONE RUN block.
# This guarantees PATH is set correctly and we can see every step's output.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && echo "=== Installing Rust ===" \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rustup-init.sh \
    && sh rustup-init.sh -y --default-toolchain stable --no-modify-path \
    && rm rustup-init.sh \
    && echo "=== Rust installed, cargo at: $(~/.cargo/bin/cargo --version) ==="

ENV ORT_STRATEGY=download

COPY rust-app/Cargo.toml rust-app/Cargo.lock* ./
COPY rust-app/src ./src

RUN echo "=== Starting cargo build ===" \
    && ~/.cargo/bin/cargo build --release \
    && echo "=== Build complete ===" \
    && ls -lh target/release/rust-app

# Hard-fail early if binary missing
RUN test -f target/release/rust-app \
    || (echo "ERROR: binary not produced" && exit 1)

# ═══════════════════════════════════════════════════════════════
# STAGE 2 — RUNNER
# ═══════════════════════════════════════════════════════════════
FROM ubuntu:24.04
WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --no-create-home --shell /bin/false appuser

COPY model_artifacts/   /app/model_artifacts/

COPY --from=builder /usr/src/app/target/release/rust-app /app/rust-app
RUN chmod +x /app/rust-app

# Copy ORT shared library so the binary can find it at runtime
COPY --from=builder /usr/src/app/target/release/build/ /app/ort-libs/
RUN find /app/ort-libs -name "libonnxruntime*.so*" -exec cp {} /app/ \; 2>/dev/null || true

USER appuser

ENV LD_LIBRARY_PATH="/app:/usr/lib"
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD bash -c 'echo -e "GET /health HTTP/1.0\r\n" | nc localhost 8000 | grep -q ok' \
    || exit 1

CMD ["./rust-app"]