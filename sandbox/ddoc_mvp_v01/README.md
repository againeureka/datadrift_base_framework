# ddoc MVP (React + FastAPI + Worker + ddoc-cli)

This is a minimal, **running** skeleton you can extend into a dataset-card EDA / compare / experiment platform.

## Services (docker-compose)
- **frontend**: React (Vite build) served on http://localhost:5173
- **api**: FastAPI on http://localhost:8000
- **worker**: Celery worker executing long-running jobs (EDA)
- **ddoc-cli**: Debug/repro CLI container (ships the same `ddoc_core`)
- **redis**: Celery broker + result backend
- **postgres**: Metadata DB (datasets)
- **minio**: Artifact store (uploaded files + reports)

## Quick start
```bash
docker compose up --build
```

Then open:
- Frontend: http://localhost:5173
- API docs: http://localhost:8000/docs
- MinIO console: http://localhost:9001 (user/pass: minioadmin / minioadmin)

## What the MVP does
1. Upload a file (csv/zip/anything) from the UI.
2. API stores it in MinIO and creates a dataset row in Postgres.
3. You can trigger **EDA** on a dataset.
4. Worker runs a simple, type-detected EDA:
   - CSV: row/col counts, missingness, basic numeric stats, top values
   - Others: file-level metadata + hash
5. Report is saved to MinIO and also returned through Celery job results.

## CLI usage (debug/repro)
In a separate terminal:
```bash
docker compose run --rm ddoc-cli ddoc --help
docker compose run --rm ddoc-cli ddoc ingest /data/sample.csv
docker compose run --rm ddoc-cli ddoc eda /data/sample.csv
```

You can mount a local folder:
```bash
docker compose run --rm -v "$PWD:/data" ddoc-cli ddoc eda /data/your.csv
```

## Notes for extension
- Add new formats by extending `ddoc_core/adapters/*` and returning **Artifacts**.
- Add new analysis via `ddoc_core/operators/*` (typed operators).
- Expose operators in API and render them as recipe steps in the UI.

