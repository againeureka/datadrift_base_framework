from __future__ import annotations
import io, os, tempfile, uuid, json
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from celery.result import AsyncResult

from .config import settings
from .db import Base, engine, get_db
from .models import Dataset
from .schemas import DatasetOut, JobOut
from .storage import put_object_bytes, presigned_get_url
from .celery_app import celery_app

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

@app.post("/datasets/{dataset_id}/eda", response_model=JobOut)
def run_eda(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    task = celery_app.send_task("worker.tasks.eda_task", args=[ds.object_key, ds.original_filename])
    return JobOut(task_id=task.id, state=task.state, result=None)

@app.get("/jobs/{task_id}", response_model=JobOut)
def job_status(task_id: str):
    r = AsyncResult(task_id, app=celery_app)
    # result will include report + report_url (if finished)
    return JobOut(task_id=task_id, state=r.state, result=r.result if r.successful() else None)
