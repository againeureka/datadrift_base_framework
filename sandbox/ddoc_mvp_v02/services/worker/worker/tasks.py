from __future__ import annotations
import os, json, tempfile, uuid
from celery import shared_task
from .storage import get_object_bytes, put_object_bytes, presigned_get_url
from ddoc_core.runtime_engine import run_operator

def _write_temp(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        return f.name

@shared_task(name="worker.tasks.run_operator_task")
def run_operator_task(operator_name: str, object_keys: list[str], filenames: list[str], params: dict):
    # download all inputs to temp files
    tmp_paths: list[str] = []
    try:
        for key, fn in zip(object_keys, filenames):
            data = get_object_bytes(key)
            suffix = os.path.splitext(fn)[1] or ""
            tmp_paths.append(_write_temp(data, suffix))

        result = run_operator(operator_name, tmp_paths, params=params or {}).model_dump()

        report_key = f"reports/{uuid.uuid4().hex}/{operator_name.replace('/', '_')}_report.json"
        put_object_bytes(
            report_key,
            json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
            "application/json",
        )
        return {
            "result": result,
            "report_object_key": report_key,
            "report_url": presigned_get_url(report_key),
        }
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass

@shared_task(name="worker.tasks.eda_task")
def eda_task(object_key: str, original_filename: str):
    # compatibility wrapper
    return run_operator_task("eda", [object_key], [original_filename], {})
