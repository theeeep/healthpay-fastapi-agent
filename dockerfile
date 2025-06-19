FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.4.9 /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-install-project --no-dev

COPY . .

RUN uv sync --frozen --no-dev

# Install your project package in editable mode to make it importable
RUN uv pip install --no-deps --editable .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
