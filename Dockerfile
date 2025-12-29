FROM python:3.11-slim

LABEL maintainer="MAPEval Team"
LABEL description="LLM-driven cryptocurrency futures trading benchmark"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    structlog \
    pydantic \
    pydantic-settings \
    pyarrow

COPY src/ ./src/
COPY tests/ ./tests/
COPY pytest.ini ./

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

RUN mkdir -p /app/logs /app/.cache

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('https://api.binance.com/api/v3/ping', timeout=5)" || exit 1

ENTRYPOINT ["python", "src/main.py"]
CMD ["--duration", "1h", "--llm-provider", "openai"]

