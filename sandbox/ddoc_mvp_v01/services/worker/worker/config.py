from __future__ import annotations
import os
from pydantic import BaseModel

class Settings(BaseModel):
    redis_url: str = os.getenv("DDOC_REDIS_URL", "redis://redis:6379/0")
    minio_endpoint: str = os.getenv("DDOC_MINIO_ENDPOINT", "minio:9000")
    minio_access_key: str = os.getenv("DDOC_MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key: str = os.getenv("DDOC_MINIO_SECRET_KEY", "minioadmin")
    minio_bucket: str = os.getenv("DDOC_MINIO_BUCKET", "ddoc")
    public_minio_url: str = os.getenv("DDOC_PUBLIC_MINIO_URL", "http://localhost:9000")

settings = Settings()
