"""``/recipes`` — site-shared recipe library (Round 19).

The library is whatever directory ``DDOC_RECIPES_DIR`` env points to,
or ``recipes/`` next to the ddoc install (the same dir that ships the
sample timeseries recipe). One YAML file = one recipe.

Endpoints:

* ``GET /recipes``               → list of `{name, path, file, description}`
* ``GET /recipes/{name}``        → the YAML text + parsed metadata + validation issues

Naming: the URL ``name`` slug is the file stem (e.g. ``timeseries_smoke``
for ``timeseries_smoke.yaml``). The recipe's own ``name:`` field is
preserved separately as ``display_name``.

Why a library? Round 17 made recipes runnable via REST + GUI; this
round adds the *catalog* layer so an operator can browse the curated
set instead of pasting YAML by memory.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_api_key

router = APIRouter(tags=["recipes"], dependencies=[Depends(require_api_key)])


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
        if p.is_file():
            recipes.append(_parse_recipe_meta(p))
    return {
        "status": "ok",
        "library_dir": str(base),
        "count": len(recipes),
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
    }
