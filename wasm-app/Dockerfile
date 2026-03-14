
# ═══════════════════════════════════════════════════════════════
# STAGE 1 — BUILDER
# ═══════════════════════════════════════════════════════════════
FROM rust:1-slim-bookworm AS builder
WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# WASM target
RUN rustup target add wasm32-wasip1

# Spin CLI
RUN curl -fsSL https://developer.fermyon.com/downloads/install.sh | bash -s -- -v v2.7.0 \
    && mv spin /usr/local/bin/

# ── Dependency-cache layer ──────────────────────────────────────
COPY wasm-app/Cargo.toml wasm-app/Cargo.lock* ./
RUN mkdir src && echo '#[allow(dead_code)] fn main() {}' > src/lib.rs
RUN spin build || true

# ── Real build ──────────────────────────────────────────────────
COPY wasm-app/src ./src
COPY wasm-app/spin.toml .
# Copy model artifacts so spin.toml file-mount resolution works at build time
COPY model_artifacts/   ./model_artifacts/
RUN touch src/lib.rs
RUN RUSTFLAGS="-C target-feature=+simd128" spin build

# ═══════════════════════════════════════════════════════════════
# STAGE 2 — RUNNER  (distroless for minimal CVE surface)
# ═══════════════════════════════════════════════════════════════
FROM gcr.io/distroless/cc-debian12
WORKDIR /app

# Spin binary
COPY --from=builder /usr/local/bin/spin /usr/local/bin/spin

# WASM binary
COPY --from=builder /usr/src/app/target/wasm32-wasip1/release/wasm_app.wasm ./wasm_app.wasm

# spin.toml references model artifacts via the files = [...] mount table;
# Spin resolves "source" paths relative to where spin.toml lives at runtime.
# We keep the same relative layout as in the builder.
COPY --from=builder /usr/src/app/spin.toml            ./spin.toml
COPY --from=builder /usr/src/app/model_artifacts/   ./model_artifacts/

EXPOSE 8000

# distroless has no shell, so HEALTHCHECK with CMD is not possible here.
# Use `docker inspect` or the Ansible health-poll task instead.

ENTRYPOINT ["/usr/local/bin/spin", "up", "--listen", "0.0.0.0:8000"]