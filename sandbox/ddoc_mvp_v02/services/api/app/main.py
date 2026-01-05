from __future__ import annotations
import io, os, tempfile, uuid, json
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from celery.result import AsyncResult

from .config import settings
from .db import Base, engine, get_db
from .models import Dataset
from .schemas import DatasetOut, JobOut, RunIn
from .storage import put_object_bytes, presigned_get_url
from .celery_app import celery_app
from ddoc_core.registry import register_builtin_operators, list_operators as core_list_operators

app = FastAPI(title="ddoc MVP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/datasets", response_model=DatasetOut)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = "dataset",
    db: Session = Depends(get_db),
):
    data = await file.read()
    object_key = f"datasets/{uuid.uuid4().hex}/{file.filename}"
    put_object_bytes(object_key, data, content_type=file.content_type or "application/octet-stream")

    ds = Dataset(name=name, original_filename=file.filename, object_key=object_key)
    db.add(ds)
    db.commit()
    db.refresh(ds)

    return DatasetOut(
        id=ds.id,
        name=ds.name,
        original_filename=ds.original_filename,
        created_at=ds.created_at,
        download_url=presigned_get_url(ds.object_key),
    )

@app.get("/datasets", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    rows = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    out = []
    for ds in rows:
        out.append(DatasetOut(
            id=ds.id,
            name=ds.name,
            original_filename=ds.original_filename,
            created_at=ds.created_at,
            download_url=presigned_get_url(ds.object_key),
        ))
    return out

@app.get("/operators")
def operators():
    register_builtin_operators()
    ops = core_list_operators()
    return [o.model_dump() for o in ops]

@app.post("/run", response_model=JobOut)
def run_operator_endpoint(run: "RunIn", db: Session = Depends(get_db)):
    if len(run.dataset_ids) == 0:
        raise HTTPException(status_code=400, detail="dataset_ids must not be empty")

    # lookup object keys
    rows = db.query(Dataset).filter(Dataset.id.in_(run.dataset_ids)).all()
    if len(rows) != len(run.dataset_ids):
        raise HTTPException(status_code=404, detail="One or more datasets not found")

    object_keys = [r.object_key for r in rows]
    filenames = [r.original_filename for r in rows]

    task = celery_app.send_task(
        "worker.tasks.run_operator_task",
        args=[run.operator_name, object_keys, filenames, run.params],
    )
    return JobOut(task_id=task.id, state=task.state, result=None)

@app.post("/datasets/{dataset_id}/eda", response_model=JobOut)
def run_eda(dataset_id: int, db: Session = Depends(get_db)):
    # backward compatible helper
    run = RunIn(operator_name="eda", dataset_ids=[dataset_id], params={})
    return run_operator_endpoint(run, db)

@app.get("/jobs/{task_id}", response_model=JobOut)
def job_status(task_id: str):
    r = AsyncResult(task_id, app=celery_app)
    return JobOut(task_id=task_id, state=r.state, result=r.result if r.successful() else None)
