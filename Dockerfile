# Torch pip wheels bundle their own CUDA libraries, so a plain Python base is enough;
# the GPU only needs the NVIDIA driver on the host and the container toolkit (Docker) or
# a CDI spec (Podman). TORCH_VARIANT=cpu builds the image for hosts without a GPU.
ARG TORCH_VARIANT=gpu

FROM ghcr.io/astral-sh/uv:0.9.7 AS uvbin

FROM python:3.12-slim-bookworm
ARG TORCH_VARIANT

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uvbin /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy

# Dependency layer: cached until pyproject.toml or uv.lock change. The cache mount keeps
# uv's wheel cache out of the image layers.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra nemo --extra ${TORCH_VARIANT}

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra nemo --extra ${TORCH_VARIANT}

# The model downloads to the Hugging Face cache under /models (a volume in compose).
# librosa's numba functions cache compiled code; the default place, next to their source
# in /app, is read-only.
ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/models \
    NUMBA_CACHE_DIR=/tmp/numba

# /app stays root-owned and read-only; the runtime user writes only /models and /tmp.
RUN mkdir -p /models && chown 1000:1000 /models
USER 1000:1000

EXPOSE 8000
# 200 once the model is loaded; the first start downloads about 2.5 GB
HEALTHCHECK --interval=15s --timeout=5s --start-period=5m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"
CMD ["uvicorn", "vadsa.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
