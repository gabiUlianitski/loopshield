FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UPSTREAM_API_BASE=https://api.openai.com \
    SHIELD_SIMILARITY_THRESHOLD=0.92 \
    SHIELD_MAX_TOKENS=8000

RUN useradd --create-home --shell /usr/sbin/nologin loopshield

COPY pyproject.toml ./
COPY loopshield ./loopshield

RUN pip install --no-cache-dir ".[proxy]"

USER loopshield

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "loopshield.proxy:app", "--host", "0.0.0.0", "--port", "8000"]
