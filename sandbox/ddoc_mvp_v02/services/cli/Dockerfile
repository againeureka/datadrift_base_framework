FROM python:3.11-slim
WORKDIR /app
COPY ddoc_core /app/ddoc_core
RUN pip install --no-cache-dir -e /app/ddoc_core
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["ddoc"]
