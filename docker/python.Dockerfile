# Executor + Kafka consumer image. Stage 1 builds the Rust sqlnorm wheel; stage 2 is a slim runtime.
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY --from=rust:1.99-slim /usr/local/cargo /usr/local/cargo
COPY --from=rust:1.99-slim /usr/local/rustup /usr/local/rustup
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo PATH=/usr/local/cargo/bin:$PATH
RUN pip install --no-cache-dir maturin==1.16.0
COPY rust/sqlnorm /src/sqlnorm
RUN maturin build --release --manifest-path /src/sqlnorm/Cargo.toml -o /wheels

FROM python:3.12-slim
COPY docker/requirements-serve.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl
COPY src /app/src
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "sqlforge.serve.executor:app", "--host", "0.0.0.0", "--port", "8000"]
