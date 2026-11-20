"""Run every drill and every Drift Zoo story through `ddoc diff` and check
the pattern reading against what each case must read.

    python zoo/eval.py            # table; exit 1 when a case misses or over-reads
    python zoo/eval.py --md       # the same as Markdown (zoo/README.md embeds it)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from drills import DRILLS, STORIES  # noqa: E402


def _diff(ref: Path, cur: Path) -> Dict[str, Any]:
    from typer.testing import CliRunner

    from ddoc.cli.main import app

    res = CliRunner().invoke(app, ["diff", str(ref), str(cur), "--json", "--lang", "en"])
    return json.loads(res.output.strip().splitlines()[-1])


def _reading(payload: Dict[str, Any]) -> Dict[str, Optional[List[str]]]:
    """Matched archetypes → their columns (None for dataset-level ones)."""
    out: Dict[str, Optional[List[str]]] = {}
    for card in payload["diagnosis"]["archetypes"]:
        if card["match"] == "match":
            out[card["code"]] = sorted(card["attrs"]) if "attrs" in card else None
    return out


def _compare(expect: Dict[str, Optional[List[str]]], got: Dict[str, Optional[List[str]]],
             strict_data: bool = True) -> List[str]:
    """Attribute patterns strictly both ways; dataset-level ones both ways
    only when ``strict_data`` (drills aimed at them, and the stories)."""
    from ddoc import archetypes as ar

    problems = [f"missed {c}" for c in expect if c not in got]
    problems += [f"extra {c}" for c in got if c not in expect
                 and (c in ar.ATTR_CODES or strict_data)]
    for code, cols in expect.items():
        if code in got and cols is not None and sorted(cols) != got[code]:
            problems.append(f"{code} on {got[code]}, expected {sorted(cols)}")
    return problems


def run() -> List[Dict[str, Any]]:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for i, drill in enumerate(DRILLS):
            ref_df, cur_df = drill.make(np.random.default_rng(1000 + i))
            d = base / drill.id
            d.mkdir()
            ref_df.to_csv(d / "ref.csv", index=False)
            cur_df.to_csv(d / "cur.csv", index=False)
            payload = _diff(d / "ref.csv", d / "cur.csv")
            got = _reading(payload)
            from ddoc import archetypes as ar
            aimed_at_data = any(c in ar.DATA_CODES for c in drill.expect)
            problems = _compare(drill.expect, got, strict_data=aimed_at_data)
            if drill.verdict and payload["verdict"] != drill.verdict:
                problems.append(f"verdict {payload['verdict']}, expected {drill.verdict}")
            rows.append({"kind": "drill", "id": drill.id, "story": drill.story, "expect": drill.expect,
                         "got": got, "verdict": payload["verdict"], "score": payload.get("overall_score"),
                         "max_attr": max((v for _, v in payload.get("top_attributes") or []), default=None),
                         "checked_verdict": bool(drill.verdict), "problems": problems})
        for case, expect in STORIES.items():
            spec = importlib.util.spec_from_file_location(f"zoo_{case}", HERE / case / "make.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            ref, cur = mod.make(base / case)
            payload = _diff(Path(ref), Path(cur))
            got = _reading(payload)
            rows.append({"kind": "story", "id": case, "story": "", "expect": expect, "got": got,
                         "verdict": payload["verdict"], "score": payload.get("overall_score"),
                         "max_attr": max((v for _, v in payload.get("top_attributes") or []), default=None),
                         "checked_verdict": False, "problems": _compare(expect, got)})
    return rows


def _titles(reading: Dict[str, Optional[List[str]]]) -> str:
    from ddoc import archetypes as ar

    return " · ".join(ar.CATALOG[c]["ko"]["title"] + (f" ({', '.join(cols)})" if cols else "")
                      for c, cols in reading.items())


def markdown(rows: List[Dict[str, Any]]) -> str:
    from ddoc import archetypes as ar

    drilled = {c for r in rows if r["kind"] == "drill" for c in r["expect"]}
    total = len(ar.DATA_CODES) + len(ar.ATTR_CODES)
    passed = sum(not r["problems"] for r in rows)
    lines = [f"케이스 {len(rows)}개 중 {passed}개가 기대한 판독과 정확히 일치합니다 "
             f"(빠진 유형도, 더 나온 유형도 없음). 유형 {total}가지 중 {len(drilled)}가지를 드릴로 확인하며, "
             "임베딩 분리형은 이미지·텍스트 플러그인이 필요해 드릴에서 뺐습니다.", "",
             "| 케이스 | 심은 변화 | 판정 (최대 속성 점수) | 판독 | 결과 |", "|---|---|---|---|---|"]
    for r in rows:
        name = f"`{r['id']}`" if r["kind"] == "drill" else f"[{r['id']}]({r['id']}/)"
        result = "✓" if not r["problems"] else "✕ " + "; ".join(r["problems"])
        score = "" if r["max_attr"] is None else f" ({r['max_attr']:.3f})"
        lines.append(f"| {name} | {r['story'] or '이야기 케이스'} | {r['verdict']}{score} | {_titles(r['got'])} | {result} |")
    quiet = [r["id"] for r in rows if r["kind"] == "drill" and not r["checked_verdict"] and r["verdict"] == "OK"]
    if quiet:
        lines += ["", f"**판정 민감도.** 변화를 심은 속성 드릴 {len(quiet)}개({', '.join(f'`{q}`' for q in quiet)})가 "
                  "판정은 OK 입니다. 판독은 변화를 정확히 짚고(잠복 변화형으로도 표시) 있지만, 분포 점수(Jensen–Shannon "
                  "발산)가 단일 속성 기준 0.25 에 닿지 않습니다. 판정 기준과 점수 방식은 이 표로 근거를 보며 따로 정할 문제라 "
                  "드릴은 판정을 확인하지 않고 그대로 보여줍니다."]
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    rows = run()
    if "--md" in argv:
        print(markdown(rows))
    else:
        for r in rows:
            mark = "ok  " if not r["problems"] else "FAIL"
            print(f"{mark} {r['id']:<20} {r['verdict']:<6} {', '.join(r['got'])}"
                  + ("" if not r["problems"] else f"   ← {'; '.join(r['problems'])}"))
    return 1 if any(r["problems"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
