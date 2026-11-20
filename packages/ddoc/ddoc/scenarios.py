"""The scenario catalog: what to run, by situation.

One file, ``scenarios.conf`` next to this module, feeds every way of
finding a command: ``ddoc scenarios`` (list, cheat sheet, run, menu),
``./run.sh menu``, the Scenarios tab of ``ddoc serve`` and the generated
``docs/cheatsheet.md``. The format is described at the top of that file.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

CATALOG = Path(__file__).with_name("scenarios.conf")
_PLACE = re.compile(r"\{(\w+)\}")
_BRACKET = re.compile(r"\[([^\]]*)\]")


@dataclass
class Param:
    name: str
    default: str
    help: Dict[str, str]


@dataclass
class Scenario:
    id: str
    text: Dict[str, Dict[str, str]] = field(default_factory=dict)   # key → {lang: value}
    tips: Dict[str, List[str]] = field(default_factory=dict)        # lang → tips
    run: List[str] = field(default_factory=list)
    params: List[Param] = field(default_factory=list)
    gui: Optional[str] = None

    def get(self, key: str, lang: str) -> str:
        values = self.text.get(key) or {}
        return values.get(lang) or values.get("en", "")

    def tip_list(self, lang: str) -> List[str]:
        return self.tips.get(lang) or self.tips.get("en", [])

    def required(self) -> List[str]:
        """Params with no default that appear outside an optional [...] group."""
        outside = " ".join(_BRACKET.sub("", line) for line in self.run)
        names = set(_PLACE.findall(outside))
        return [p.name for p in self.params if not p.default and p.name in names]


class CatalogError(ValueError):
    pass


def load(path: Path = CATALOG) -> List[Scenario]:
    scenarios: List[Scenario] = []
    current: Optional[Scenario] = None
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"\[([a-z0-9-]+)\]", line)
        if m:
            current = Scenario(m.group(1))
            scenarios.append(current)
            continue
        if current is None or "=" not in line:
            raise CatalogError(f"{path.name}:{n}: expected [id] or key = value")
        key, value = (s.strip() for s in line.split("=", 1))
        base, _, lang = key.partition(".")
        lang = lang or "en"
        if base == "run":
            current.run.append(value)
        elif base == "tip":
            current.tips.setdefault(lang, []).append(value)
        elif base == "param":
            parts = [s.strip() for s in value.split("|")]
            if len(parts) != 4:
                raise CatalogError(f"{path.name}:{n}: param = name | default | help | help (ko)")
            current.params.append(Param(parts[0], parts[1], {"en": parts[2], "ko": parts[3]}))
        elif base == "gui":
            current.gui = value
        elif base in ("group", "title", "when"):
            current.text.setdefault(base, {})[lang] = value
        else:
            raise CatalogError(f"{path.name}:{n}: unknown key {key!r}")
    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise CatalogError("duplicate scenario id")
    return scenarios


def find(scenario_id: str, scenarios: Optional[List[Scenario]] = None) -> Scenario:
    for s in scenarios if scenarios is not None else load():
        if s.id == scenario_id:
            return s
    raise CatalogError(f"no scenario {scenario_id!r} (ddoc scenarios lists them)")


def values(scenario: Scenario, given: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    out = {p.name: p.default for p in scenario.params}
    out.update({k: v for k, v in (given or {}).items() if v is not None})
    return out


def expand(line: str, vals: Dict[str, str]) -> List[str]:
    """One run line → argument list. A bracketed group disappears when any
    placeholder in it is empty; every value stays one argument."""
    def bracket(m: "re.Match[str]") -> str:
        inner = m.group(1)
        return inner if all(vals.get(name) for name in _PLACE.findall(inner)) else ""

    args = []
    for token in shlex.split(_BRACKET.sub(bracket, line)):
        if _PLACE.fullmatch(token):
            value = vals.get(token[1:-1], "")
            if value:
                args.append(value)
        else:
            args.append(_PLACE.sub(lambda m: vals.get(m.group(1), ""), token))
    return args


def commands(scenario: Scenario, given: Optional[Dict[str, str]] = None) -> List[List[str]]:
    vals = values(scenario, given)
    missing = [name for name in scenario.required() if not vals.get(name)]
    if missing:
        raise CatalogError("needs: " + ", ".join(f"--set {m}=…" for m in missing))
    return [expand(line, vals) for line in scenario.run]


def display(scenario: Scenario, lang: str, given: Optional[Dict[str, str]] = None) -> List[str]:
    """Run lines as a person would type them, with <name> for missing values."""
    vals = values(scenario, given)
    for name in scenario.required():
        vals.setdefault(name, "")
        if not vals[name]:
            vals[name] = f"<{name}>"
    return ["ddoc " + " ".join(shlex.quote(a) if not a.startswith("<") else a for a in expand(line, vals))
            for line in scenario.run]


def fill(text: str, scenario: Scenario, given: Optional[Dict[str, str]] = None) -> str:
    vals = values(scenario, given)
    return _PLACE.sub(lambda m: vals.get(m.group(1)) or m.group(0), text)


def to_dict(scenario: Scenario, lang: str) -> Dict[str, Any]:
    return {"id": scenario.id, "group": scenario.get("group", lang), "title": scenario.get("title", lang),
            "when": scenario.get("when", lang), "commands": display(scenario, lang),
            "tips": [fill(tip, scenario) for tip in scenario.tip_list(lang)], "gui": scenario.gui,
            "params": [{"name": p.name, "default": p.default, "help": p.help.get(lang) or p.help["en"],
                        "required": p.name in scenario.required()} for p in scenario.params]}


def cheatsheet(lang: str = "ko") -> str:
    """The catalog as Markdown, grouped as in the file."""
    head = {"ko": "상황마다 그대로 칠 명령입니다. `ddoc scenarios run <id> --set 이름=값` 으로 바로 실행하고, "
                  "`-n` 을 붙이면 실행하지 않고 명령만 보여줍니다. `./run.sh menu` 는 번호로 골라 실행합니다.",
            "en": "The command for each situation. `ddoc scenarios run <id> --set name=value` runs it; `-n` "
                  "only prints it. `./run.sh menu` lets you pick by number."}
    lines = [head.get(lang, head["en"]), ""]
    group = None
    for s in load():
        g = s.get("group", lang)
        if g != group:
            group = g
            if lines[-1] != "":
                lines.append("")
            lines += [f"### {g}", ""]
        lines.append(f"- **{s.get('title', lang)}** (`{s.id}`) — {s.get('when', lang)}  ")
        for cmd in display(s, lang):
            lines.append(f"  `{cmd}`  ")
        for tip in s.tip_list(lang):
            lines.append(f"  - {fill(tip, s)}")
    return "\n".join(lines)
