from __future__ import annotations
from pydantic import BaseModel
import os

class Settings(BaseModel):
    database_url: str = os.getenv("DDOC_DATABASE_URL", "sqlite:///./ddoc.db")
    redis_url: str = os.getenv("DDOC_REDIS_URL", "redis://localhost:6379/0")

    minio_endpoint: str = os.getenv("DDOC_MINIO_ENDPOINT", "localhost:9000")
    minio_access_key: str = os.getenv("DDOC_MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key: str = os.getenv("DDOC_MINIO_SECRET_KEY", "minioadmin")
    minio_bucket: str = os.getenv("DDOC_MINIO_BUCKET", "ddoc")
    public_minio_url: str = os.getenv("DDOC_PUBLIC_MINIO_URL", "http://localhost:9000")

    cors_origins: str = os.getenv("DDOC_CORS_ORIGINS", "http://localhost:5173")

settings = Settings()
