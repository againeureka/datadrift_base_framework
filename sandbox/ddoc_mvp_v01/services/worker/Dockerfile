FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
 && rm -rf /var/lib/apt/lists/*

COPY ddoc_core /app/ddoc_core
RUN pip install --no-cache-dir -e /app/ddoc_core

COPY services/worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/worker/worker /app/worker

ENV PYTHONUNBUFFERED=1
CMD ["celery", "-A", "worker.celery_worker", "worker", "--loglevel=INFO", "--concurrency=1"]
