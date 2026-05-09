"""``/recipes`` — site-shared recipe library (Round 19 → Round 21).

The library is whatever directory ``DDOC_RECIPES_DIR`` env points to,
or ``recipes/`` next to the ddoc install. One YAML file = one recipe.

Endpoints:

* ``GET    /recipes``                       → list `{name, file, display_name, step_count}`
* ``GET    /recipes/{name}``                → YAML text + parsed metadata + validation issues
* ``PUT    /recipes/{name}``                → save / update YAML (Round 20, write-mode only)
* ``DELETE /recipes/{name}``                → soft-delete (auto-archive) (Round 21)
* ``POST   /recipes/{name}/restore/{ts}``   → restore active from a snapshot (Round 21)
* ``GET    /recipes/{name}/diff``           → unified diff between two refs (Round 21)
* ``GET    /recipes/{name}/versions``       → snapshot history under .history/ (Round 20)
* ``GET    /recipes/{name}/versions/{ts}``  → one historical YAML

Write-mode is opt-in via ``DDOC_RECIPES_WRITE=1`` env so a casual
``ddoc serve`` deployment cannot have its recipes overwritten by
unauthenticated callers (combined with ``DDOC_API_KEY`` for actual
production access control).

Versioning: every successful PUT/DELETE/restore also writes
``<library>/.history/<name>/<UTC ts>.yaml`` so the previous content is
recoverable. The active recipe stays at ``<library>/<name>.yaml``;
``versions`` listing is most-recent-first.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from ..auth import require_api_key

router = APIRouter(tags=["recipes"], dependencies=[Depends(require_api_key)])


# Allow only conservative filename characters in URL slugs so write
# routes can't traverse outside the library.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _write_enabled() -> bool:
    return os.getenv("DDOC_RECIPES_WRITE", "").lower() in ("1", "true", "yes")


def _check_write_enabled() -> None:
    if not _write_enabled():
        raise HTTPException(
            status_code=403,
            detail={
                "status": "error",
                "error_code": "library_read_only",
                "message": "Recipe library is read-only. Set DDOC_RECIPES_WRITE=1 to enable save/versioning routes.",
            },
        )


def _check_safe_name(name: str) -> None:
    if not _SAFE_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "error_code": "invalid_recipe_name",
                "message": "name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
                "name": name,
            },
        )


def _resolve_library_dir() -> Optional[Path]:
    """Return the recipe library directory, or None if not configured."""
    env_dir = os.getenv("DDOC_RECIPES_DIR")
    if env_dir:
        d = Path(env_dir)
        return d if d.exists() else None
    # Fall back to the bundled ``recipes/`` next to the package — works
    # for editable installs (monorepo dev) and for site installs that
    # symlink a curated dir there.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "recipes"
        if candidate.exists() and any(candidate.glob("*.yaml")):
            return candidate
    return None


def _safe_join(base: Path, name: str) -> Path:
    """Resolve ``name.yaml`` (or ``.yml``) under ``base`` and refuse
    anything outside the library."""
    for suffix in (".yaml", ".yml"):
        candidate = (base / f"{name}{suffix}").resolve()
        if candidate.is_file() and base.resolve() in candidate.parents:
            return candidate
    raise FileNotFoundError(name)


def _parse_recipe_meta(path: Path) -> Dict[str, Any]:
    """Extract a recipe's ``name`` / ``description`` / step count
    without running validation. Returns shape suitable for /recipes
    listing."""
    try:
        import yaml
    except ImportError:
        return {"file": path.name, "error": "yaml_missing"}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return {"file": path.name, "error": f"parse_failed: {e}"}
    if not isinstance(data, dict):
        return {"file": path.name, "error": "top_level_not_mapping"}
    steps = data.get("steps") or []
    return {
        "name": path.stem,
        "file": path.name,
        "display_name": data.get("name"),
        "description": data.get("description"),
        "step_count": len(steps) if isinstance(steps, list) else 0,
    }


@router.get("/recipes")
def list_recipes() -> Dict[str, Any]:
    """List every YAML recipe in the configured library."""
    base = _resolve_library_dir()
    if base is None:
        return {
            "status": "ok",
            "library_dir": None,
            "count": 0,
            "recipes": [],
            "hint": (
                "Set DDOC_RECIPES_DIR env to a directory of recipe YAML files, "
                "or place them under <package_root>/recipes/."
            ),
        }
    recipes: List[Dict[str, Any]] = []
    for p in sorted(base.glob("*.yaml")) + sorted(base.glob("*.yml")):
        # Skip the .history archive — it's only reachable via
        # /recipes/{name}/versions endpoints.
        if p.is_file() and ".history" not in p.parts:
            recipes.append(_parse_recipe_meta(p))
    return {
        "status": "ok",
        "library_dir": str(base),
        "count": len(recipes),
        "write_enabled": _write_enabled(),
        "recipes": recipes,
    }


@router.get("/recipes/{name}")
def get_recipe(name: str) -> Dict[str, Any]:
    """Return the YAML text + parsed metadata + validation issues."""
    base = _resolve_library_dir()
    if base is None:
        raise HTTPException(
            status_code=404,
            detail={"status": "error", "error_code": "library_not_configured",
                    "message": "DDOC_RECIPES_DIR not set and no bundled recipes/ found."},
        )
    try:
        path = _safe_join(base, name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={"status": "error", "error_code": "recipe_not_found", "name": name},
        )

    yaml_text = path.read_text(encoding="utf-8")
    meta = _parse_recipe_meta(path)
    issues: List[str] = []
    try:
        from ddoc.core.recipe import Recipe
        recipe = Recipe.load(path)
        issues = recipe.validate()
    except Exception as e:  # noqa: BLE001
        issues = [f"load_failed: {e}"]
    return {
        "status": "ok",
        "name": name,
        "path": str(path),
        "yaml": yaml_text,
        "metadata": meta,
        "issues": issues,
        "write_enabled": _write_enabled(),
    }


# ── Round 20 — write + versioning ───────────────────────────────────


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _archive_existing(base: Path, name: str) -> Optional[str]:
    """Snapshot the current active recipe to .history/<name>/<ts>.yaml
    before overwriting. Returns the archive path on success, or None
    when there's nothing to archive (new file).

    Timestamp granularity is one second; if multiple archives land in
    the same second (back-to-back saves, restore-then-save), append a
    ``-N`` suffix to keep each snapshot recoverable.
    """
    try:
        existing = _safe_join(base, name)
    except FileNotFoundError:
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history_dir = base / ".history" / name
    history_dir.mkdir(parents=True, exist_ok=True)
    archive = history_dir / f"{ts}.yaml"
    n = 1
    while archive.exists():
        archive = history_dir / f"{ts}-{n}.yaml"
        n += 1
    archive.write_text(existing.read_text(encoding="utf-8"), encoding="utf-8")
    return str(archive)


@router.put("/recipes/{name}")
def save_recipe(name: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Save (create or update) a recipe.

    Body: ``{yaml: "..."}``. The YAML is parsed + validated before
    writing; write fails (no archive, no overwrite) if validation
    surfaces structural issues. The previous content (if any) is
    snapshotted under ``.history/<name>/<UTC ts>.yaml``.

    Requires ``DDOC_RECIPES_WRITE=1`` env. URL ``name`` must match
    a conservative slug pattern.
    """
    _check_write_enabled()
    _check_safe_name(name)

    yaml_text = payload.get("yaml")
    if not isinstance(yaml_text, str) or not yaml_text.strip():
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "error_code": "missing_yaml",
                    "message": "Body must be {'yaml': <non-empty string>}."},
        )

    base = _resolve_library_dir()
    if base is None:
        raise HTTPException(
            status_code=404,
            detail={"status": "error", "error_code": "library_not_configured",
                    "message": "DDOC_RECIPES_DIR not set and no bundled recipes/ found."},
        )
    base = base.resolve()

    # Validate via the recipe loader before writing.
    import tempfile
    from ddoc.core.recipe import Recipe, RecipeError
    tmp_path = Path(tempfile.mkstemp(prefix="ddoc-recipe-save-", suffix=".yaml")[1])
    tmp_path.write_text(yaml_text, encoding="utf-8")
    try:
        recipe = Recipe.load(tmp_path)
    except RecipeError as e:
        return {"status": "error", "error_code": e.code, "message": str(e)}
    finally:
        try:
            tmp_path.unlink()
        except Exception:  # noqa: BLE001
            pass
    issues = recipe.validate()
    if issues:
        return {"status": "error", "error_code": "validation_failed", "issues": issues}

    target = base / f"{name}.yaml"
    archive_path = _archive_existing(base, name)
    _atomic_write(target, yaml_text)
    return {
        "status": "ok",
        "name": name,
        "path": str(target),
        "archived_to": archive_path,
        "step_count": len(recipe.steps),
    }


@router.get("/recipes/{name}/versions")
def list_versions(name: str) -> Dict[str, Any]:
    """List archived snapshots of a recipe (most-recent first)."""
    _check_safe_name(name)
    base = _resolve_library_dir()
    if base is None:
        raise HTTPException(
            status_code=404,
            detail={"status": "error", "error_code": "library_not_configured"},
        )
    history_dir = base / ".history" / name
    if not history_dir.exists():
        return {"status": "ok", "name": name, "count": 0, "versions": []}
    versions = sorted(
        (p for p in history_dir.glob("*.yaml") if p.is_file()),
        key=lambda p: p.stem,
        reverse=True,
    )
    return {
        "status": "ok",
        "name": name,
        "count": len(versions),
        "versions": [
            {"timestamp": p.stem, "size_bytes": p.stat().st_size, "file": p.name}
            for p in versions
        ],
    }


@router.get("/recipes/{name}/versions/{ts}")
def get_version(name: str, ts: str) -> Dict[str, Any]:
    """Return the YAML text of a specific historical snapshot."""
    _check_safe_name(name)
    if not _SAFE_NAME_RE.match(ts):
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "error_code": "invalid_timestamp"},
        )
    base = _resolve_library_dir()
    if base is None:
        raise HTTPException(
            status_code=404,
            detail={"status": "error", "error_code": "library_not_configured"},
        )
    candidate = (base / ".history" / name / f"{ts}.yaml").resolve()
    history_root = (base / ".history" / name).resolve()
    if history_root not in candidate.parents or not candidate.is_file():
        raise HTTPException(
            status_code=404,
            detail={"status": "error", "error_code": "version_not_found",
                    "name": name, "timestamp": ts},
        )
    return {
        "status": "ok",
        "name": name,
        "timestamp": ts,
        "yaml": candidate.read_text(encoding="utf-8"),
        "size_bytes": candidate.stat().st_size,
    }


# ── Round 21 — delete / restore / diff ──────────────────────────────


def _resolve_ref_yaml(base: Path, name: str, ref: str) -> str:
    """Resolve a ref ('HEAD' or a snapshot timestamp) to YAML text."""
    if ref == "HEAD":
        try:
            active = _safe_join(base, name)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "error_code": "recipe_not_found",
                        "name": name, "ref": ref},
            )
        return active.read_text(encoding="utf-8")
    if not _SAFE_NAME_RE.match(ref):
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "error_code": "invalid_ref", "ref": ref},
        )
    candidate = (base / ".history" / name / f"{ref}.yaml").resolve()
    history_root = (base / ".history" / name).resolve()
    if history_root not in candidate.parents or not candidate.is_file():
        raise HTTPException(
            status_code=404,
            detail={"status": "error", "error_code": "version_not_found",
                    "name": name, "ref": ref},
        )
    return candidate.read_text(encoding="utf-8")


@router.delete("/recipes/{name}")
def delete_recipe(name: str) -> Dict[str, Any]:
    """Delete a recipe. The current content is archived first so a
    subsequent ``POST /recipes/{name}/restore/{ts}`` can recover it.

    Requires ``DDOC_RECIPES_WRITE=1``.
    """
    _check_write_enabled()
    _check_safe_name(name)
    base = _resolve_library_dir()
    if base is None:
        raise HTTPException(
            status_code=404,
            detail={"status": "error", "error_code": "library_not_configured"},
        )
    base = base.resolve()
    try:
        active = _safe_join(base, name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={"status": "error", "error_code": "recipe_not_found", "name": name},
        )
    archive_path = _archive_existing(base, name)
    active.unlink()
    return {
        "status": "ok",
        "name": name,
        "deleted_path": str(active),
        "archived_to": archive_path,
    }


@router.post("/recipes/{name}/restore/{ts}")
def restore_version(name: str, ts: str) -> Dict[str, Any]:
    """Replace the active recipe with the YAML from a historical
    snapshot. The current active content is archived first so the
    restore itself is reversible.

    Requires ``DDOC_RECIPES_WRITE=1``. ``ts`` must match a file under
    ``.history/<name>/``.
    """
    _check_write_enabled()
    _check_safe_name(name)
    if not _SAFE_NAME_RE.match(ts):
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "error_code": "invalid_timestamp"},
        )
    base = _resolve_library_dir()
    if base is None:
        raise HTTPException(
            status_code=404,
            detail={"status": "error", "error_code": "library_not_configured"},
        )
    base = base.resolve()

    snapshot = (base / ".history" / name / f"{ts}.yaml").resolve()
    history_root = (base / ".history" / name).resolve()
    if history_root not in snapshot.parents or not snapshot.is_file():
        raise HTTPException(
            status_code=404,
            detail={"status": "error", "error_code": "version_not_found",
                    "name": name, "timestamp": ts},
        )
    yaml_text = snapshot.read_text(encoding="utf-8")
    target = base / f"{name}.yaml"
    archive_path = _archive_existing(base, name)
    _atomic_write(target, yaml_text)
    return {
        "status": "ok",
        "name": name,
        "restored_from": str(snapshot),
        "archived_to": archive_path,
        "path": str(target),
    }


@router.get("/recipes/{name}/diff")
def diff_recipe(
    name: str,
    from_: str = Query("HEAD", alias="from"),
    to: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Unified-diff two refs of a recipe.

    Each ref is either ``HEAD`` (current active content) or a snapshot
    timestamp under ``.history/<name>/``. Defaults: ``from=HEAD``,
    ``to=`` most-recent snapshot — convenient for "what changed since
    last save?".
    """
    _check_safe_name(name)
    base = _resolve_library_dir()
    if base is None:
        raise HTTPException(
            status_code=404,
            detail={"status": "error", "error_code": "library_not_configured"},
        )
    base = base.resolve()

    if to is None:
        # Default: compare HEAD against the most-recent snapshot.
        history_dir = base / ".history" / name
        if not history_dir.exists():
            return {
                "status": "ok",
                "name": name,
                "from": from_,
                "to": None,
                "diff": "",
                "note": "no snapshots yet",
            }
        snaps = sorted((p for p in history_dir.glob("*.yaml") if p.is_file()),
                       key=lambda p: p.stem, reverse=True)
        if not snaps:
            return {
                "status": "ok",
                "name": name,
                "from": from_,
                "to": None,
                "diff": "",
                "note": "no snapshots yet",
            }
        to = snaps[0].stem

    a_text = _resolve_ref_yaml(base, name, from_)
    b_text = _resolve_ref_yaml(base, name, to)

    import difflib
    diff_lines = list(difflib.unified_diff(
        a_text.splitlines(keepends=True),
        b_text.splitlines(keepends=True),
        fromfile=f"{name}@{from_}",
        tofile=f"{name}@{to}",
        n=3,
    ))
    return {
        "status": "ok",
        "name": name,
        "from": from_,
        "to": to,
        "diff": "".join(diff_lines),
        "identical": not diff_lines,
    }
