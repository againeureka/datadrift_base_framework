from __future__ import annotations
import os, json, tempfile, uuid
from celery import shared_task
from .storage import get_object_bytes, put_object_bytes, presigned_get_url
from ddoc_core.operators.eda import eda

@shared_task(name="worker.tasks.eda_task")
def eda_task(object_key: str, original_filename: str):
    # download to temp file
    data = get_object_bytes(object_key)
    suffix = os.path.splitext(original_filename)[1] or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        tmp_path = f.name

    try:
        report = eda(tmp_path).model_dump()
        report_key = f"reports/{uuid.uuid4().hex}/eda_report.json"
        put_object_bytes(report_key, json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")
        return {
            "report": report,
            "report_object_key": report_key,
            "report_url": presigned_get_url(report_key),
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
