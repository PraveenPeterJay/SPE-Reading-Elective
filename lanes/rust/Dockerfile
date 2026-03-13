# ── Rust inference lane ───────────────────────────────────────────────────────
# Multi-stage: build in rust:1.78-slim, run in debian:bookworm-slim
# ort 2.x with load-dynamic: downloads ORT shared lib at build time via ort-sys
# ─────────────────────────────────────────────────────────────────────────────

FROM rust:latest AS builder

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev cmake curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY lanes/rust/Cargo.toml ./Cargo.toml
COPY lanes/rust/src         ./src

# ort 2.x load-dynamic: fetch ORT shared library during build
ENV ORT_DYLIB_PATH=/build/libonnxruntime.so
RUN curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/\
onnxruntime-linux-x64-1.17.3.tgz \
    | tar -xz --strip-components=2 -C /build \
      onnxruntime-linux-x64-1.17.3/lib/libonnxruntime.so.1.17.3 && \
    mv /build/libonnxruntime.so.1.17.3 /build/libonnxruntime.so

ENV RUSTFLAGS="-C target-cpu=native"
RUN cargo build --release

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/spe-rust-lane /app/spe-rust-lane
COPY --from=builder /build/libonnxruntime.so            /usr/lib/libonnxruntime.so

COPY model_artifacts/model.onnx  /app/model.onnx
COPY model_artifacts/scaler.json /app/scaler.json

RUN useradd -r -s /bin/false appuser
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -fs http://localhost:8501/health || exit 1

ENV ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
CMD ["/app/spe-rust-lane"]