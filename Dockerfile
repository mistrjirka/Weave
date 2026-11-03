# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci
COPY frontend ./frontend
COPY tsconfig.json vite.config.ts ./
RUN npm run build

FROM python:3.12-slim-bookworm AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ARG WEAVE_UID=1000
ARG WEAVE_GID=1000
RUN groupadd --gid "$WEAVE_GID" weave \
    && useradd --uid "$WEAVE_UID" --gid weave --create-home weave \
    && mkdir -p /workspace/state /workspace/projects /workspace/settings \
    && chown -R weave:weave /workspace
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    WEAVE_HOST=0.0.0.0 WEAVE_PORT=8173 \
    WEAVE_CONFIG_FILE=/workspace/settings/weave.json \
    WEAVE_DATA=/workspace/state WEAVE_PROJECTS_DIR=/workspace/projects
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --python /usr/local/bin/python
USER weave
EXPOSE 8173
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8173/api/health', timeout=3)"]
CMD ["/app/.venv/bin/python", "-m", "backend.installation", "run"]

# Opt in to training with --target training. Select a torch wheel/index for
# your hardware rather than assuming the newest CUDA build supports every GPU.
FROM base AS training-deps
USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
ARG TORCH_PACKAGE=torch
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY training-requirements.txt ./
RUN uv venv --python /usr/local/bin/python /opt/training \
    && uv pip install --python /opt/training/bin/python "$TORCH_PACKAGE" --index-url "$TORCH_INDEX_URL" \
    && uv pip install --python /opt/training/bin/python -r training-requirements.txt \
    && /opt/training/bin/python -c "import torch, transformers, pyarrow, jsonschema; print(torch.__version__)"
ENV WEAVE_TRAINING_ENABLED=1 WEAVE_TRAINING_PYTHON=/opt/training/bin/python
USER weave

FROM base AS web
COPY --chown=weave:weave backend ./backend
COPY --chown=weave:weave references ./references
COPY --chown=weave:weave examples ./examples
COPY --chown=weave:weave LICENSE ./
COPY --chown=weave:weave *.md ./
COPY --chown=weave:weave docs ./docs
COPY --chown=weave:weave --from=frontend /build/dist ./dist
RUN mkdir -p /home/weave/.cache

FROM training-deps AS training
COPY --chown=weave:weave backend ./backend
COPY --chown=weave:weave references ./references
COPY --chown=weave:weave examples ./examples
COPY --chown=weave:weave LICENSE ./
COPY --chown=weave:weave *.md ./
COPY --chown=weave:weave docs ./docs
COPY --chown=weave:weave --from=frontend /build/dist ./dist
RUN mkdir -p /home/weave/.cache

# A plain docker build creates the smaller editor/export image.
FROM web AS editor
ENV WEAVE_TRAINING_ENABLED=0
