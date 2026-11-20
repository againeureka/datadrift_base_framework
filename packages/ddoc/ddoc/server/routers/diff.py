"""``POST /diff`` — wraps ``ddoc diff``: verdict, diagnosis and report.

The verdict is an answer, not an error: OK / DRIFT / BLIND all return
HTTP 200 with the CLI's JSON payload (``exit_code`` 0 / 1 / 2 alongside).
With ``include_report`` the self-contained HTML diagnostic report comes
back inline as ``report_html``, so the GUI can show it without a shared
filesystem.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from ..auth import require_api_key
from ..runner import run
from ..schemas import DiffRequest

router = APIRouter(tags=["diff"], dependencies=[Depends(require_api_key)])


def _argv(req: DiffRequest, report: str | None) -> List[str]:
    args = ["diff", req.ref, req.cur, "--json",
            "--threshold", str(req.threshold), "--attr-threshold", str(req.attr_threshold)]
    if req.detector and req.detector != "default":
        args += ["--detector", req.detector]
    if req.lang:
        args += ["--lang", req.lang]
    if req.issuer:
        args += ["--issuer", req.issuer]
    if req.sample_rows:
        args += ["--sample-rows", str(req.sample_rows)]
    if req.with_embeddings:
        args.append("--with-embeddings")
    if report:
        args += ["--report", report]
    return args


@router.post("/diff")
def diff(req: DiffRequest) -> Dict[str, Any]:
    """Judge CUR against REF. Returns the ``ddoc diff --json`` payload
    plus ``exit_code`` and, when asked, ``report_html``."""
    tmpdir = tempfile.mkdtemp(prefix="ddoc-diff-")
    try:
        report = str(Path(tmpdir) / "report.html") if req.include_report else None
        result = run(_argv(req, report), require_json=True, timeout=req.timeout_sec,
                     ok_returncodes=(0, 1, 2))
        payload: Dict[str, Any] = dict(result.json or {})
        payload["exit_code"] = result.returncode
        # A server temp path means nothing to the caller: drop it and the
        # "report written to …" hint that names it.
        path = payload.pop("report_path", None)
        if path:
            payload["hints"] = [h for h in payload.get("hints") or [] if path not in h]
        if report and Path(report).exists():
            payload["report_html"] = Path(report).read_text(encoding="utf-8")
        return payload
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
