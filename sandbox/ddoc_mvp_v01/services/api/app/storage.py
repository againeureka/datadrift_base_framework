from __future__ import annotations
import boto3
from botocore.client import Config
from .config import settings

def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=f"http://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

def put_object_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    c = s3_client()
    c.put_object(Bucket=settings.minio_bucket, Key=key, Body=data, ContentType=content_type)

def presigned_get_url(key: str, expires: int = 3600) -> str:
    c = s3_client()
    return c.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.minio_bucket, "Key": key},
        ExpiresIn=expires,
    )
