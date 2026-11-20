"""Keep the parts of the docs that come from code in step with the code.

    python scripts/docs.py build     # rewrite every generated block
    python scripts/docs.py check     # exit 1 when a block is stale (the tests run this)
    python scripts/docs.py shots     # re-take the README screenshots (needs Chrome + Pillow)

A generated block sits in a Markdown file between

    <!-- generated:NAME — scripts/docs.py build -->
    ...
    <!-- /generated:NAME -->

and is replaced as a whole. Everything outside the markers is hand-written.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Tuple

PKG = Path(__file__).resolve().parent.parent          # packages/ddoc
ROOT = PKG.parent.parent                               # repository root
DOCS = PKG / "docs"

sys.path.insert(0, str(PKG))


# ── Generated blocks ──────────────────────────────────────────────────


def _archetypes() -> str:
    from ddoc import archetypes as ar

    rows = ["| 유형 | 범위 | 판단 기준 |", "|---|---|---|"]
    for scope, codes in (("데이터 전체", ar.DATA_CODES), ("속성", ar.ATTR_CODES)):
        for code in codes:
            rows.append(f"| {ar.CATALOG[code]['ko']['title']} | {scope} | {ar.criterion_text(code, 'ko')} |")
    total = len(ar.DATA_CODES) + len(ar.ATTR_CODES)
    return (f"현재 {total}가지입니다 (σ = 기준 데이터의 표준편차, %p = 퍼센트포인트). "
            f"\"가까움\"은 모자란 조건 하나가 기준의 {ar.limits()['near']} 이상일 때입니다.\n\n"
            + "\n".join(rows))


def _profile_findings() -> str:
    from ddoc import profile as pf

    rows = ["| 소견 | 범위 | 판단 기준 |", "|---|---|---|"]
    for scope, codes in (("데이터 전체", pf.DATA_CODES), ("열", pf.COLUMN_CODES)):
        for code in codes:
            rows.append(f"| {pf.CATALOG[code]['ko']['title']} | {scope} | {pf.criterion_text(code, 'ko')} |")
    return f"현재 {len(pf.CATALOG)}가지입니다.\n\n" + "\n".join(rows)


def _cheatsheet() -> str:
    from ddoc import scenarios

    return scenarios.cheatsheet("ko")


BLOCKS: Dict[str, Tuple[Path, Callable[[], str]]] = {
    "archetypes": (DOCS / "tutorial" / "concepts.md", _archetypes),
    "profile-findings": (DOCS / "guides" / "analysis.md", _profile_findings),
    "cheatsheet": (DOCS / "cheatsheet.md", _cheatsheet),
}

_MARK = re.compile(r"(<!-- generated:(?P<name>[\w-]+)[^>]*-->\n)(?P<body>.*?)(<!-- /generated:(?P=name) -->)", re.S)


def _render(path: Path) -> Tuple[str, List[str]]:
    """The file with its blocks regenerated, and the names it contains."""
    text = path.read_text(encoding="utf-8")
    names: List[str] = []

    def sub(m: "re.Match[str]") -> str:
        name = m.group("name")
        names.append(name)
        if name not in BLOCKS:
            raise SystemExit(f"{path}: unknown generated block {name!r}")
        return m.group(1) + BLOCKS[name][1]().rstrip("\n") + "\n" + m.group(4)

    return _MARK.sub(sub, text), names


def _files() -> Dict[Path, None]:
    return dict.fromkeys(path for path, _ in BLOCKS.values())


def build() -> int:
    for path in _files():
        new, names = _render(path)
        if new != path.read_text(encoding="utf-8"):
            path.write_text(new, encoding="utf-8")
            print(f"updated {path.relative_to(ROOT)} ({', '.join(names)})")
    return 0


def stale() -> List[str]:
    """Blocks whose text differs from what the code would generate, or
    that are missing from their file."""
    out: List[str] = []
    for path in _files():
        new, names = _render(path)
        for name, (target, _) in BLOCKS.items():
            if target == path and name not in names:
                out.append(f"{path.relative_to(ROOT)}: block {name!r} missing")
        if new != path.read_text(encoding="utf-8"):
            out.append(f"{path.relative_to(ROOT)}: out of date — run scripts/docs.py build")
    return out


def check() -> int:
    problems = stale()
    for p in problems:
        print(p)
    return 1 if problems else 0


# ── Screenshots ───────────────────────────────────────────────────────

_PROBE = """<script>window.addEventListener('load', function () {
  var out = {};
  function box(el) { var r = el.getBoundingClientRect(); return [Math.round(r.top + scrollY), Math.round(r.bottom + scrollY)]; }
  document.querySelectorAll('h2').forEach(function (h) { out['h2:' + h.textContent.replace(/^\\d+\\.\\s*/, '')] = box(h); });
  [['.cards .card', 'card'], ['.cards', 'cards'], ['.callouts', 'callouts'], ['.dg', 'dg']].forEach(function (p) {
    var el = document.querySelector(p[0]); if (el) out[p[1]] = box(el); });
  document.body.setAttribute('data-pos', JSON.stringify(out));
});</script></body>"""


def _chrome() -> str:
    for c in (os.environ.get("DDOC_CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser")):
        if c and Path(c).exists():
            return c
    raise SystemExit("Chrome not found (set DDOC_CHROME)")


def _profile_demo(path: Path) -> None:
    """A small order table with the quality issues the profile looks for."""
    import random

    rng = random.Random(3)
    rows = ["order_id,order_date,price,channel,qty,grade,coupon"]
    for i in range(300):
        price = 900.0 if i % 40 == 0 else rng.gauss(130, 10)
        channel = rng.choice(["store", "online", "online", "app", "Store"])
        grade = "" if i % 4 == 0 else rng.choice("AB")
        rows.append(f"{1000 + i},2026-{7 + i // 100:02d}-{1 + i % 28:02d},{price:.1f},{channel},"
                    f"{rng.randint(1, 4)},{grade},{rng.choice('YN')}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _render_page(work: Path, chrome: str, html_path: Path, name: str) -> Tuple[Path, Dict[str, List[int]]]:
    page = html_path.read_text(encoding="utf-8")
    page = re.sub(r"<html lang='(\w+)'>", r"<html lang='\1' data-theme='light'>", page, count=1)
    light = work / f"{name}.html"
    light.write_text(page, encoding="utf-8")
    probe = work / f"{name}.probe.html"
    probe.write_text(page.replace("</body>", _PROBE, 1), encoding="utf-8")
    prof = work / "chrome-profile"
    base = [chrome, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}", "--no-first-run",
            "--disable-crash-reporter", "--disable-breakpad", "--hide-scrollbars", "--window-size=1100,5200"]
    dom = subprocess.run(base + ["--dump-dom", probe.as_uri()], capture_output=True, text=True).stdout
    pos = json.loads(re.search(r'data-pos="([^"]*)"', dom).group(1).replace("&quot;", '"'))
    shot = work / f"{name}.png"
    subprocess.run(base + [f"--screenshot={shot}", light.as_uri()], capture_output=True)
    return shot, pos


def shots() -> int:
    from PIL import Image

    from ddoc.i18n import t

    chrome = _chrome()
    ddoc = [sys.executable, "-m", "ddoc.cli.main"]
    zoo = PKG / "zoo" / "regime_shift"
    if not (zoo / "data" / "ref.csv").exists():
        subprocess.run([sys.executable, "make.py"], cwd=zoo, check=True)
    out_dir = ROOT / "docs" / "imgs"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # Relative paths only: the report prints them, and screenshots go public.
        for lang in ("en", "ko"):
            subprocess.run(ddoc + ["diff", "data/ref.csv", "data/cur.csv", "--lang", lang,
                                   "--report", str(work / f"zoo_{lang}.html")], cwd=zoo, capture_output=True)
        _profile_demo(work / "orders.csv")
        eda = subprocess.run(ddoc + ["analyze", "eda", "--data-path", "orders.csv", "--json", "--quiet"],
                             cwd=work, capture_output=True, text=True).stdout.strip().splitlines()[-1]
        (work / "eda.json").write_text(eda, encoding="utf-8")
        subprocess.run(ddoc + ["report", "render", "-i", "eda.json", "-o", "profile_en.html", "--lang", "en"],
                       cwd=work, capture_output=True, check=True)

        zoo_en, p_en = _render_page(work, chrome, work / "zoo_en.html", "zoo_en_l")
        zoo_ko, p_ko = _render_page(work, chrome, work / "zoo_ko.html", "zoo_ko_l")
        prof, p_pr = _render_page(work, chrome, work / "profile_en.html", "profile_l")

        def h2(pos: Dict[str, List[int]], key: str, lang: str) -> int:
            return pos[f"h2:{t(key, lang)}"][0]

        jobs = [
            (zoo_en, 0, p_en["card"][1] + 10, "diagnostic-report.png"),
            (zoo_en, h2(p_en, "sec.diagram", "en") - 20, p_en["callouts"][1] + 10, "data-diagram.png"),
            (prof, h2(p_pr, "sec.summary", "en") - 20, p_pr["callouts"][1] + 10, "profile-report.png"),
            (zoo_ko, h2(p_ko, "sec.reading", "ko") - 20, p_ko["cards"][1] + 10, "regime-shift-reading-ko.png"),
        ]
        for src, y0, y1, name in jobs:
            im = Image.open(src).convert("RGB").crop((0, max(0, y0), 1100, y1))
            im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
                out_dir / name, optimize=True)
            print(f"wrote docs/imgs/{name} {im.size[0]}x{im.size[1]}")
    return 0


def main(argv: List[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "check"
    if cmd not in ("build", "check", "shots"):
        print(__doc__)
        return 2
    return {"build": build, "check": check, "shots": shots}[cmd]()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
