"""What the GUI's front page needs besides ``POST /diff``.

Choosing data without typing paths, and seeing it before judging it:

* ``GET  /fs/list``            browse folders under the data roots
* ``GET  /fs/image``           one image under the data roots (previews)
* ``POST /workspace/uploads``  open an upload session
* ``PUT  /workspace/uploads/{id}/{side}/{name}``  one dropped file (raw body)
* ``GET  /cases``, ``POST /cases/{id}``  example cases with a known answer
* ``POST /describe``           what one path is: kind, size, columns, period
* ``POST /probe``              what ``ddoc diff`` would do with a pair

Browsing is confined to the data roots (``DDOC_DATA_ROOT``, several
separated like PATH; by default the folder the server was started in and
the home folder). Uploads and built cases go to a workspace folder
(``DDOC_WORKSPACE``, by default under the system temp folder) whose
sessions are removed after a day. Nothing here judges data or runs a
command: that stays with ``POST /diff``.
"""
from __future__ import annotations

import os
import re
import secrets
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..auth import require_api_key

router = APIRouter(tags=["workbench"], dependencies=[Depends(require_api_key)])

MAX_ENTRIES = 1000
SESSION_TTL_SEC = 24 * 3600
_SESSION = re.compile(r"^(up|case)-[0-9a-f]{12}$")
_SIDES = ("ref", "cur")


def _fail(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"status": "error", "error_code": code, "message": message})


# ── Places ────────────────────────────────────────────────────────────


def workspace_root() -> Path:
    return Path(os.getenv("DDOC_WORKSPACE") or Path(tempfile.gettempdir()) / "ddoc-workspace").resolve()


def data_roots() -> List[Path]:
    given = [p for p in (os.getenv("DDOC_DATA_ROOT") or "").split(os.pathsep) if p.strip()]
    candidates = [Path(p).expanduser() for p in given] or [Path.cwd(), Path.home()]
    roots: List[Path] = []
    for p in candidates:
        try:
            r = p.resolve()
        except OSError:
            continue
        if r.is_dir() and r not in roots:
            roots.append(r)
    return roots


def _inside(path: Path, roots: List[Path]) -> bool:
    return any(path == r or r in path.parents for r in roots)


def _browsable(raw: str) -> Path:
    """``raw`` resolved, refused unless it is under a data root or the workspace."""
    try:
        path = Path(raw).expanduser().resolve()
    except OSError:
        raise _fail(400, "bad_path", f"cannot resolve {raw!r}")
    if not _inside(path, data_roots() + [workspace_root()]):
        raise _fail(403, "outside_data_roots",
                    "That folder is outside the data roots. Start the server there or set DDOC_DATA_ROOT.")
    return path


def _new_session(prefix: str) -> Path:
    root = workspace_root()
    root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for old in root.iterdir():           # yesterday's sessions
        try:
            if _SESSION.match(old.name) and now - old.stat().st_mtime > SESSION_TTL_SEC:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    session = root / f"{prefix}-{secrets.token_hex(6)}"
    session.mkdir()
    return session


def _kind(path: Path) -> str:
    from ddoc.cli.commands.diff import _AUDIO_EXTS, _IMAGE_EXTS, _TEXT_EXTS

    ext = path.suffix.lower()
    return ("csv" if ext == ".csv" else "image" if ext in _IMAGE_EXTS else "audio" if ext in _AUDIO_EXTS
            else "text" if ext in _TEXT_EXTS else "other")


# ── Browse ────────────────────────────────────────────────────────────


@router.get("/fs/list")
def fs_list(path: Optional[str] = None) -> Dict[str, Any]:
    """Folders and files of ``path`` (the data roots when omitted)."""
    roots = data_roots()
    if not path:
        return {"path": None, "parent": None, "roots": [str(r) for r in roots], "truncated": False,
                "entries": [{"name": str(r), "path": str(r), "type": "dir", "kind": "dir"} for r in roots],
                "counts": {}}
    here = _browsable(path)
    if not here.is_dir():
        raise _fail(404, "not_a_folder", f"{here} is not a folder")
    entries: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    try:
        children = sorted(here.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as e:
        raise _fail(403, "unreadable", str(e))
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
            kind = "dir" if is_dir else _kind(child)
            counts[kind] = counts.get(kind, 0) + 1
            if len(entries) < MAX_ENTRIES:
                entries.append({"name": child.name, "path": str(child), "type": "dir" if is_dir else "file",
                                "kind": kind, "size": None if is_dir else child.stat().st_size})
        except OSError:
            continue
    parent = here.parent if _inside(here.parent, roots + [workspace_root()]) else None
    return {"path": str(here), "parent": str(parent) if parent else None, "roots": [str(r) for r in roots],
            "entries": entries, "counts": counts, "truncated": sum(counts.values()) > len(entries)}


@router.get("/fs/image")
def fs_image(path: str):
    """One image file under the data roots, for previews."""
    target = _browsable(path)
    if not target.is_file() or _kind(target) != "image":
        raise _fail(404, "not_an_image", "not an image file")
    return FileResponse(str(target))


# ── Uploads ───────────────────────────────────────────────────────────


@router.post("/workspace/uploads")
def open_upload() -> Dict[str, Any]:
    session = _new_session("up")
    for side in _SIDES:
        (session / side).mkdir()
    return {"id": session.name, **{side: str(session / side) for side in _SIDES}}


@router.put("/workspace/uploads/{session_id}/{side}/{name:path}")
async def upload_file(session_id: str, side: str, name: str, request: Request) -> Dict[str, Any]:
    """Store one file (the raw request body) on a side of an upload session.
    ``name`` may carry folders (a dropped folder keeps its layout)."""
    base = workspace_root() / session_id
    if not _SESSION.match(session_id) or side not in _SIDES or not (base / side).is_dir():
        raise _fail(404, "no_such_upload", "unknown upload session")
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    if not parts or any(p in (".", "..") or p.startswith(".") for p in parts):
        raise _fail(400, "bad_name", "file names must not be hidden or contain '..'")
    dest = (base / side).joinpath(*parts)
    if (base / side).resolve() not in dest.resolve().parents:
        raise _fail(400, "bad_name", "file name leaves the upload folder")
    limit = int(float(os.getenv("DDOC_UPLOAD_MAX_MB", "1024")) * 1024 * 1024)
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with open(dest, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise _fail(413, "too_large", f"file exceeds {limit // (1024 * 1024)} MB (DDOC_UPLOAD_MAX_MB)")
                f.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    return {"path": str(base / side), "file": str(dest), "size": size}


# ── Example cases ─────────────────────────────────────────────────────


@router.get("/cases")
def list_cases(lang: Optional[str] = None) -> List[Dict[str, str]]:
    from ddoc.examples import cases
    from ddoc.i18n import resolve_lang

    return [cases.to_dict(c, resolve_lang(lang)) for c in cases.CASES]


@router.post("/cases/{case_id}")
def build_case(case_id: str) -> Dict[str, str]:
    """Write the case's two CSV files into the workspace; returns their paths."""
    from ddoc.examples import cases

    if case_id not in cases.BY_ID:
        raise _fail(404, "no_such_case", f"unknown case {case_id!r}")
    ref, cur = cases.build(case_id, _new_session("case"))
    return {"id": case_id, "ref": str(ref), "cur": str(cur)}


# ── Look before judging ───────────────────────────────────────────────


class PathBody(BaseModel):
    path: str
    lang: Optional[str] = None


class PairBody(BaseModel):
    ref: str
    cur: Optional[str] = None
    lang: Optional[str] = None


@router.post("/describe")
def describe(body: PathBody) -> Dict[str, Any]:
    """What one path holds: kind of data, how much, and for CSV the columns
    and the period (from a sample; the row count is the total)."""
    from ddoc.cli.commands.diff import _as_datetime, _load_csv_sampled, _profile, sniff_modality
    from ddoc.cli.commands.doctor import _describe

    path = Path(body.path).expanduser()
    info = _describe(path)
    if not info["exists"]:
        return info
    info["name"] = path.name
    if info["kind"] == "csv":
        files = sniff_modality(path)[1].get("files") or [str(path)]
        try:
            frame, total = _load_csv_sampled(files, 20_000)
            times = {c: p for c in frame.columns if (p := _as_datetime(frame[c])) is not None}
            profile = _profile(frame, files, times, 0)
            profile["rows"] = int(total)
            info["profile"] = profile
        except Exception as e:  # noqa: BLE001 — an unreadable CSV is an answer, not a crash
            info["error"] = str(e)
    elif info["kind"] == "vision" and path.is_dir():
        from ddoc.cli.commands.diff import _IMAGE_EXTS

        info["images"] = [str(p) for p in sorted(path.rglob("*"))
                          if p.suffix.lower() in _IMAGE_EXTS and not p.name.startswith(".")][:6]
    return info


@router.post("/probe")
def probe_pair(body: PairBody) -> Dict[str, Any]:
    """What ``ddoc diff`` would do with these paths, without judging."""
    from ddoc.cli.commands.doctor import probe
    from ddoc.i18n import resolve_lang

    try:
        return probe(Path(body.ref).expanduser(), Path(body.cur).expanduser() if body.cur else None,
                     resolve_lang(body.lang))
    except Exception as e:  # noqa: BLE001
        raise _fail(400, "probe_failed", str(e))
