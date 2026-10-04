# Serve-only image for the public demo (F2). No Docling, no PyTorch, no spaCy: the index is
# built offline and travels as a release asset, so this image only has to answer questions.
#
#   docker build -f deploy/Dockerfile --build-arg INDEX_URL=<release asset url> -t amr .
#   docker run --rm -p 8000:8000 --memory=512m -e BYOK_ONLY=true -e APP_ENV=prod amr
#
# Two things are baked in at build time so a cold start does not reach the network:
#   * the index (downloaded, verified against its own manifest, unpacked to /app/data/index)
#   * the bge-small ONNX weights, which fastembed would otherwise fetch on the first question
#
# The image holds no credentials. `BYOK_ONLY=true` makes that explicit: every request must
# carry the caller's own provider key, and the server has no quota of its own to spend.

FROM python:3.13-slim AS build

ARG INDEX_URL
ARG INDEX_SHA256=""
WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# The index. A bad or truncated download must fail the build, not the first request.
RUN test -n "$INDEX_URL" || (echo "INDEX_URL build-arg is required" >&2; exit 1) \
 && curl -fsSL --retry 3 -o index.tar.gz "$INDEX_URL" \
 && if [ -n "$INDEX_SHA256" ]; then echo "$INDEX_SHA256  index.tar.gz" | sha256sum -c -; fi \
 && mkdir -p /build/index \
 && tar -xzf index.tar.gz -C /build/index \
 && test -f /build/index/manifest.json \
 && rm index.tar.gz


FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

# fastembed keeps the ONNX weights here; baked in below so no question waits on a download
ENV FASTEMBED_CACHE_PATH=/opt/fastembed HF_HUB_DISABLE_TELEMETRY=1

# one worker, one thread: a 512 MB instance has no room for a second ONNX arena
ENV OMP_NUM_THREADS=1

# The deployment contract. `BYOK_ONLY` means the server has no quota of its own to spend;
# `PUBLIC_DEPLOY` turns off the loopback-only request guard, because here the platform's proxy
# is the boundary. Startup refuses to serve if these two ever disagree (`check_public_safety`).
ENV APP_ENV=prod BYOK_ONLY=true PUBLIC_DEPLOY=true SERVER_HOST=0.0.0.0

WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# Bake the embedding weights. Same alias the index was built with (`thresholds.retrieval.embedder`);
# a mismatch is caught by `check_index` at startup rather than silently answering worse.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY config/ ./config/
COPY frontend/ ./frontend/
RUN pip install --no-cache-dir -e . --no-deps

COPY --from=build /build/index /app/data/index
RUN mkdir -p /app/data/cache /app/data/logs

# Render sets $PORT; default to 8000 for a local `docker run`. `SERVER_HOST` feeds app.yaml,
# so the startup check sees the address the server actually binds (see DEPLOY.md).
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "exec uvicorn rag.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 65"]
