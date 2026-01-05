FROM python:3.11-slim

WORKDIR /app

# system deps (psycopg, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
 && rm -rf /var/lib/apt/lists/*

COPY ddoc_core /app/ddoc_core
RUN pip install --no-cache-dir -e /app/ddoc_core

COPY services/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/api/app /app/app

ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
