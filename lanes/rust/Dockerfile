# ── Rust inference lane ───────────────────────────────────────────────────────
# Multi-stage: build in rust:1.78-slim, run in distroless/cc
# Minimal attack surface: no shell, no package manager at runtime.
# ─────────────────────────────────────────────────────────────────────────────

FROM rust:1.78-slim AS builder

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev cmake curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY lanes/rust/Cargo.toml ./Cargo.toml
COPY lanes/rust/src         ./src

# Download ORT shared library
ENV ORT_LIB_LOCATION=/build/ort_lib
RUN mkdir -p $ORT_LIB_LOCATION && \
    curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/\
onnxruntime-linux-x64-1.17.3.tgz \
    | tar -xz -C $ORT_LIB_LOCATION --strip-components=1

ENV RUSTFLAGS="-C target-cpu=native"
RUN cargo build --release

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/spe-rust-lane /app/spe-rust-lane
COPY --from=builder /build/ort_lib/lib/libonnxruntime.so.1.17.3 /usr/lib/libonnxruntime.so.1.17.3
RUN ln -s /usr/lib/libonnxruntime.so.1.17.3 /usr/lib/libonnxruntime.so

COPY model_artifacts/model.onnx  /app/model.onnx
COPY model_artifacts/scaler.json /app/scaler.json

RUN useradd -r -s /bin/false appuser
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fs http://localhost:8501/health || exit 1

CMD ["/app/spe-rust-lane"]