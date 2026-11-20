"""``GET /scenarios`` — the scenario catalog for the GUI's Scenarios tab.

Read-only: it lists what to run by situation. Running happens through the
existing endpoints (a scenario with ``gui`` opens that tab pre-filled), so
this adds no new way to execute commands on the server.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends

from ..auth import require_api_key

router = APIRouter(tags=["scenarios"], dependencies=[Depends(require_api_key)])


@router.get("/scenarios")
def scenarios(lang: Optional[str] = None) -> List[Dict[str, Any]]:
    from ddoc import scenarios as sc
    from ddoc.i18n import resolve_lang

    lang = resolve_lang(lang)
    return [sc.to_dict(s, lang) for s in sc.load()]
