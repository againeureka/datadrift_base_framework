"""ddoc.archetypes — each written-up pattern is read from data that has it."""
from __future__ import annotations

import random
import re
import string

import pytest

from ddoc import archetypes as ar
from ddoc import diagnosis as dx
from ddoc.cli.commands.diff import compute_verdict, csv_diff


def _write(path, header, rows):
    path.write_text(header + "\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n")
    return str(path)


def _read(tmp_path, header, ref_rows, cur_rows):
    ref = _write(tmp_path / "r.csv", header, ref_rows)
    cur = _write(tmp_path / "c.csv", header.split("|")[-1], cur_rows) if "|" in header else \
        _write(tmp_path / "c.csv", header, cur_rows)
    result = csv_diff([ref], [cur])
    verdict, _, top = compute_verdict(result, threshold=0.15, attr_threshold=0.25)
    d = dx.build(verdict=verdict, overall=result["overall_score"], top=top, threshold=0.15,
                 attr_threshold=0.25, result=result, evidence=result["distributions"])
    for card in d["archetypes"]:  # every card renders in both languages
        for lang in ("en", "ko"):
            text = ar.card_text(card, lang, profile=result.get("profile"))
            assert text["title"] and text["signals"]
    return verdict, d["archetypes"]


def _codes(cards, match="match"):
    return {(c["code"], tuple(c.get("attrs") or ())) for c in cards if c["match"] == match}


rng = random.Random(5)
N = 400


def _gauss(mu, sd, n=N):
    return [round(rng.gauss(mu, sd), 3) for _ in range(n)]


def test_unit_change_hides_the_general_numeric_patterns(tmp_path):
    base = _gauss(100, 10)
    _, cards = _read(tmp_path, "amount", [[x] for x in base], [[round(x * 1000, 1)] for x in _gauss(100, 10)])
    codes = _codes(cards)
    assert ("unit_change", ("amount",)) in codes
    assert not {c for c, _ in codes} & {"level_shift", "spread_change", "shift_and_spread", "tail_outliers"}
    unit = next(c for c in cards if c["code"] == "unit_change")
    assert unit["signals"][0]["factor"] == 1000
    assert "×1000" in ar.card_text(unit, "ko")["signals"][0]


def test_level_shift_and_spread_change_and_tail_outliers(tmp_path):
    header = "level,spread,tail,id"
    ref = list(zip(_gauss(50, 5), _gauss(50, 5), _gauss(50, 5), range(N)))
    tail = _gauss(50, 5)
    for i in range(0, N, 12):
        tail[i] = 200.0
    cur = list(zip(_gauss(55, 5), _gauss(50, 12), tail, range(N)))
    _, cards = _read(tmp_path, header, ref, cur)
    codes = {c for c, _ in _codes(cards)}
    assert {"level_shift", "spread_change", "tail_outliers"} <= codes
    by = {c["code"]: set(c["attrs"]) for c in cards if c["match"] == "match" and "attrs" in c}
    assert by["level_shift"] == {"level"}
    assert by["spread_change"] == {"spread"} and by["tail_outliers"] == {"tail"}
    assert "shift_and_spread" not in by  # neither a wider spread nor spikes alone is a move


def test_shift_with_narrower_spread(tmp_path):
    # moved and narrowed (×0.5, and ×0.75 where the spread alone would be "about the
    # same"), against narrowed in place
    header = "narrow,mild,squeeze,id"
    ref = list(zip(_gauss(50, 5), _gauss(50, 5), _gauss(50, 5), range(N)))
    cur = list(zip(_gauss(55, 2.5), _gauss(55, 3.75), _gauss(50, 2.5), range(N)))
    _, cards = _read(tmp_path, header, ref, cur)
    by = {c["code"]: set(c["attrs"]) for c in cards if c["match"] == "match" and "attrs" in c}
    assert by["shift_and_narrow"] == {"narrow", "mild"}
    assert by["spread_change"] == {"squeeze"}
    assert not {"level_shift", "shift_and_spread"} & set(by)
    card = next(c for c in cards if c["code"] == "shift_and_narrow")
    line = ar.card_text(card, "ko")["signals"][0]
    assert re.fullmatch(r"narrow: 평균 [\d.]+ → [\d.]+ \([\d.]+σ\), 표준편차 [\d.]+ → [\d.]+ \(×0\.\d\d\)", line)


def test_stuck_values_numeric_and_categorical(tmp_path):
    ref = [[x, rng.choice("AB")] for x in _gauss(10, 2)]
    cur = [[7.0, "A" if i % 50 else "B"] for i in range(N)]
    _, cards = _read(tmp_path, "temp,state", ref, cur)
    stuck = next(c for c in cards if c["code"] == "stuck_values" and c["match"] == "match")
    assert set(stuck["attrs"]) == {"temp", "state"}


def test_missing_surge_new_vanished_and_spelling_variant(tmp_path):
    ref = [[rng.choice(["store", "online", "phone"]), "x"] for _ in range(N)]
    cur = [[rng.choice(["Store", "online", "kiosk"]), "" if i % 4 == 0 else "x"] for i in range(N)]
    _, cards = _read(tmp_path, "channel,flag", ref, cur)
    codes = {c for c, _ in _codes(cards)}
    assert {"new_category", "vanished_category", "missing_surge"} <= codes
    new = next(c for c in cards if c["code"] == "new_category")
    assert ["Store", "store"] in new["signals"][0]["variants"]
    assert "Store ≈ store" in ar.card_text(new, "ko")["signals"][0]


def test_mix_shift_mentions_the_period(tmp_path):
    ref = [[f"2026-01-{1 + i % 28:02d}", "store" if i % 4 else "online"] for i in range(N)]
    cur = [[f"2026-07-{1 + i % 28:02d}", "online" if i % 4 else "store"] for i in range(N)]
    _, cards = _read(tmp_path, "order_date,channel", ref, cur)
    mix = next(c for c in cards if c["code"] == "mix_shift")
    result = csv_diff([str(tmp_path / "r.csv")], [str(tmp_path / "c.csv")])
    lines = ar.card_text(mix, "ko", profile=result["profile"])["signals"]
    assert any(line.startswith("기간: 2026-01-01") for line in lines)


def test_regime_local_schema_and_small_sample(tmp_path):
    letters = string.ascii_lowercase[:4]
    ref = [[rng.choice(letters) for _ in range(4)] for _ in range(20)]
    cur = [[rng.choice("wxyz"), *r[1:]] for r in ref]
    ref_rows = [r + ["only"] for r in ref]
    path_r = _write(tmp_path / "r.csv", "a,b,c,d,legacy", ref_rows)
    path_c = _write(tmp_path / "c.csv", "a,b,c,d", cur)
    result = csv_diff([path_r], [path_c])
    verdict, _, top = compute_verdict(result, threshold=0.15, attr_threshold=0.25)
    d = dx.build(verdict=verdict, overall=result["overall_score"], top=top, threshold=0.15,
                 attr_threshold=0.25, result=result, evidence=result["distributions"])
    codes = {c["code"] for c in d["archetypes"] if c["match"] == "match"}
    assert {"schema_change", "small_sample", "local_shift"} <= codes and "regime_shift" not in codes


def test_hidden_change_is_a_pattern_within_sampling_noise(tmp_path):
    # a 1–5 rating whose mean drops 0.8σ: on 30 rows a side chance could do
    # that (pattern only near, verdict OK, hidden change); on 400 it cannot
    small_ref = [[v] for v in [1] * 2 + [2] * 3 + [3] * 7 + [4] * 13 + [5] * 5]
    small_cur = [[v] for v in [1] * 8 + [2] * 7 + [3] * 6 + [4] * 5 + [5] * 4]
    verdict, cards = _read(tmp_path, "rating", small_ref, small_cur)
    assert verdict == "OK"
    assert ("hidden_change", ()) in _codes(cards) and "stable" not in {c for c, _ in _codes(cards)}
    assert ("shift_and_spread", ("rating",)) in _codes(cards, "near")
    verdict, cards = _read(tmp_path, "rating", small_ref * 13, small_cur * 13)
    assert verdict == "DRIFT" and ("shift_and_spread", ("rating",)) in _codes(cards)


def test_a_pattern_beyond_noise_drifts_even_when_scores_stay_low(tmp_path):
    """A few far outliers barely move a 20-bin distribution score, yet they
    are a finding: a confirmed pattern makes the verdict DRIFT, and the
    rationale says why."""
    base = _gauss(50, 5)
    tail = list(base)
    for i in range(0, N, 20):
        tail[i] = 500.0                                   # 5 % far outside the baseline range
    calm_ref, calm_cur = _gauss(10, 2), _gauss(10, 2)     # a column that did not change
    verdict, cards = _read(tmp_path, "x,y", list(zip(base, calm_ref)), list(zip(tail, calm_cur)))
    assert verdict == "DRIFT" and ("tail_outliers", ("x",)) in _codes(cards)
    result = csv_diff([str(tmp_path / "r.csv")], [str(tmp_path / "c.csv")])
    assert result["attribute_drifts"]["x"] < 0.25 and result["overall_score"] < 0.15
    _, _, top = compute_verdict(result, threshold=0.15, attr_threshold=0.25)
    d = dx.build(verdict="DRIFT", overall=result["overall_score"], top=top, threshold=0.15,
                 attr_threshold=0.25, result=result, evidence=result["distributions"])
    assert [r["code"] for r in d["rationale"]] == ["pattern"]
    assert "극단값 출현형" in dx.rationale_text(d["rationale"][0], "ko")


def test_near_is_an_early_signal_not_a_match(tmp_path):
    base = _gauss(50, 10)
    sd = (sum((x - sum(base) / N) ** 2 for x in base) / N) ** 0.5
    ref = [[x] for x in base]
    cur = [[round(x + 0.43 * sd, 3)] for x in base]  # exactly 0.43σ: under 0.5σ, over 0.4σ
    _, cards = _read(tmp_path, "x", ref, cur)
    assert ("level_shift", ("x",)) in _codes(cards, "near")
    assert ("level_shift", ("x",)) not in _codes(cards)


@pytest.mark.parametrize("code", sorted(ar.CATALOG))
def test_catalog_entries_agree_across_languages(code):
    en, ko = ar.CATALOG[code]["en"], ar.CATALOG[code]["ko"]
    assert set(en) == set(ko)
    assert set(re.findall(r"{(\w+)}", en["signal"])) == set(re.findall(r"{(\w+)}", ko["signal"]))
    assert len(en["causes"]) == len(ko["causes"]) and len(en["actions"]) == len(ko["actions"])


def test_every_code_has_a_write_up():
    assert set(ar.DATA_CODES) | set(ar.ATTR_CODES) == set(ar.CATALOG)
