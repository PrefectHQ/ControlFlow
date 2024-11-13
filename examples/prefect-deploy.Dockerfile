FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y git

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_SYSTEM_PYTHON=1
ENV PATH="/root/.local/bin:$PATH"

RUN uv pip install controlflow

