FROM python:3.11-slim AS runtime

LABEL maintainer="MAPEval Team"
LABEL description="LLM-driven cryptocurrency futures trading benchmark"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m pip install --no-cache-dir '.[api,db,full]' \
    && adduser --disabled-password --gecos '' appuser \
    && mkdir -p /app/logs /app/.cache \
    && chown -R appuser:appuser /app/logs /app/.cache

USER appuser

ENTRYPOINT ["python", "-m", "mapeval"]
CMD ["--non-interactive", "--execution-mode", "simulation", "--duration", "1h", "--llm-provider", "openai"]


FROM runtime AS test

USER root
COPY tests/ ./tests/
COPY pytest.ini ./
RUN python -m pip install --no-cache-dir '.[dev]'
USER appuser


FROM runtime AS final
