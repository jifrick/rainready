FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr for Cloud Run structured logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source only (.dockerignore excludes .env and agent skills)
COPY main.py .
COPY static ./static

# Run as non-root user per Google Cloud Run security best practices
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Cloud Run Container Runtime Contract:
# - Listen on 0.0.0.0 (never 127.0.0.1) and honor injected $PORT (default 8080)
# - Use `exec` so uvicorn runs as PID 1 and handles SIGTERM cleanly on scale-to-zero
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*'"]
