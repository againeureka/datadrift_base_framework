"""Drift archetypes — measured signals read against written-up patterns.

Drift tends to come in recognisable shapes: a whole regime moves, one
column changes unit, a sensor degrades, a new category appears, a field
starts arriving empty. Each archetype here pairs *conditions* on numbers
the comparison already measured with a short write-up: what the shape
usually means, common causes, what to do first, and what not to conclude.

The data decides which write-ups apply. When every condition holds the
archetype is a ``match``; when all but one hold and that one is close to
its threshold (within :data:`NEAR`), it is ``near`` — an early signal
rather than a finding. Output is language-neutral (codes and numbers);
:func:`card_text` renders it through :data:`CATALOG`.

The write-ups are general knowledge, not a diagnosis of this data; the
"first actions" are how to confirm or rule out the cause.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# Pattern thresholds are the diagnosis's own, so the two never disagree.
from .diagnosis import (MEAN_SHIFT_STD, MISSING_CHANGE, NEW_VALUE_SHARE, OUT_OF_RANGE,
                        SHARE_SHIFT, SMALL_SAMPLE, SPREAD_RATIO)

NEAR = 0.8                 # "near": the missing condition is ≥ 80 % of its threshold
REGIME_SHARE = 0.5         # ≥ half of the attributes over the threshold
LOCAL_SHARE = 1 / 3        # ≤ a third of the attributes over the threshold
EMBED_SEPARATION = 0.5     # embedding neighbour mixing ≤ 0.5 (1 − mixing ≥ 0.5)
UNIT_RATIO = 2.0           # mean moved by at least ×2 (or ÷2) ...
UNIT_COHERENCE = 0.2       # ... and the std by the same factor (|ln(mean/std ratio)| ≤ 0.2)
BULK_STEADY = 1.25         # the middle half kept its width within ×1.25
SPREAD_WITH_SHIFT = 1.25   # spread growth that counts when the values also moved
STUCK_SHARE = 0.95         # one value holds ≥ 95 % of the current data ...
STUCK_REF_SHARE = 0.8      # ... where the baseline's top value held ≤ 80 %
_FACTORS = (10, 100, 1000, 60, 3600, 24)


DATA_CODES = ("schema_change", "small_sample", "embedding_shift", "regime_shift", "local_shift",
              "hidden_change", "stable")
ATTR_CODES = ("unit_change", "stuck_values", "shift_and_spread", "level_shift", "spread_change",
              "tail_outliers", "missing_surge", "new_category", "vanished_category", "mix_shift")


# ── Measuring ─────────────────────────────────────────────────────────


def _shares(counts: Dict[str, float]) -> Dict[str, float]:
    total = sum(v for v in counts.values() if isinstance(v, (int, float)))
    return {k: v / total for k, v in counts.items() if isinstance(v, (int, float))} if total else {}


def _measure(summary: Dict[str, Any], dists: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    m: Dict[str, Any] = {"kind": summary.get("kind")}
    missing = summary.get("missing") or {}
    if missing:
        m["missing_ref"], m["missing_cur"] = missing.get("ref") or 0.0, missing.get("cur") or 0.0
        m["missing_delta"] = m["missing_cur"] - m["missing_ref"]
    if m["kind"] == "numeric":
        mean, std = summary.get("mean") or {}, summary.get("std") or {}
        mr, mc, sr, sc = mean.get("ref"), mean.get("cur"), std.get("ref"), std.get("cur")
        m.update(mean_ref=mr, mean_cur=mc, std_ref=sr, std_cur=sc,
                 out=summary.get("outside_ref_range") or 0.0)
        if mr is not None and mc is not None:
            m["shift"] = abs(mc - mr) / sr if sr else (0.0 if mc == mr else math.inf)
        if sr and sc is not None:
            m["spread"] = sc / sr
            m["spread_dist"] = (sc / sr if sc >= sr else (sr / sc if sc else math.inf))
        if mr and mc and sr and sc and (mr > 0) == (mc > 0):
            m["mean_ratio"] = mc / mr
        # Robust to outliers: where the middle of the data sits and how wide it is.
        med, q1, q3 = summary.get("median") or {}, summary.get("q1") or {}, summary.get("q3") or {}
        if sr and med.get("ref") is not None and med.get("cur") is not None:
            m["median_shift"] = abs(med["cur"] - med["ref"]) / sr
        if None not in (q1.get("ref"), q3.get("ref"), q1.get("cur"), q3.get("cur")):
            ref_iqr, cur_iqr = q3["ref"] - q1["ref"], q3["cur"] - q1["cur"]
            if ref_iqr > 0 and cur_iqr > 0:
                m["iqr_ratio"] = cur_iqr / ref_iqr
    elif m["kind"] == "categorical":
        m["new"] = summary.get("new_values") or []
        m["vanished"] = summary.get("vanished_values") or []
        m["new_mass"] = sum(v[1] for v in m["new"])
        m["vanished_mass"] = sum(v[1] for v in m["vanished"])
        m["variants"] = summary.get("variants") or []
        mover = summary.get("top_mover")
        if mover:
            m["mover"], m["share_delta"] = mover, abs(mover[2] - mover[1])
        if dists:
            rs, cs = _shares(dists.get("ref") or {}), _shares(dists.get("cur") or {})
            rs.pop("(other)", None)
            cs.pop("(other)", None)
            if rs and cs:
                top = max(cs, key=cs.get)
                m.update(top_label=top, top_cur=cs[top], top_ref=max(rs.values()))
    return m


def _status(conds: List[Tuple[float, float]]) -> Optional[str]:
    """``conds`` = (value, threshold) pairs, each read as value ≥ threshold.
    All met → match; all but one, that one ≥ NEAR × threshold → near."""
    missed = [(v, th) for v, th in conds if not v >= th]
    if not missed:
        return "match"
    if len(missed) == 1 and missed[0][1] > 0 and missed[0][0] >= NEAR * missed[0][1]:
        return "near"
    return None


def _nearest_factor(ratio: float) -> Optional[float]:
    for f in _FACTORS:
        for target in (f, 1 / f):
            if abs(ratio / target - 1) <= 0.05:
                return target
    return None


def _attr_cards(name: str, m: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Archetypes for one attribute, most specific first; a specific
    match (unit change, degradation) hides the general ones it explains."""
    out: List[Dict[str, Any]] = []
    explained: set = set()

    def add(code: str, status: Optional[str], **signal: Any) -> None:
        if status and code not in explained:
            out.append({"code": code, "match": status, "attr": name, "signal": signal})

    if m["kind"] == "numeric":
        ratio, spread = m.get("mean_ratio"), m.get("spread")
        if ratio and spread:
            coherent = abs(math.log(ratio / spread)) <= UNIT_COHERENCE
            big = max(ratio, 1 / ratio)
            if coherent and big >= UNIT_RATIO:
                add("unit_change", "match", mean_ref=m["mean_ref"], mean_cur=m["mean_cur"],
                    ratio=ratio, spread=spread, factor=_nearest_factor(ratio))
                explained |= {"level_shift", "spread_change", "shift_and_spread", "tail_outliers"}
        if m.get("std_ref") and m.get("std_cur") == 0:
            add("stuck_values", "match", value=m.get("mean_cur"), share=1.0)
            explained |= {"level_shift", "spread_change", "shift_and_spread", "tail_outliers"}
        shift, dist, out_share = m.get("shift") or 0.0, m.get("spread_dist") or 1.0, m.get("out", 0.0)
        # Outliers in the tails: the middle of the data stayed (median and
        # interquartile width), yet values beyond the baseline range showed
        # up. Large spikes also move the mean and std; those are explained.
        iqr = m.get("iqr_ratio")
        bulk_steady = (m.get("median_shift", math.inf) < MEAN_SHIFT_STD and iqr is not None
                       and 1 / BULK_STEADY <= iqr <= BULK_STEADY)
        if bulk_steady:
            st = _status([(out_share, OUT_OF_RANGE)])
            add("tail_outliers", st, out=out_share)
            if st == "match":
                explained |= {"level_shift", "spread_change", "shift_and_spread"}
        grew = (spread or 1.0) > 1
        if grew:
            # two signals backing each other: a lower bar for the spread.
            # (Values outside the baseline range do not count as a move:
            # a wider spread produces them by itself.)
            st = _status([(dist, SPREAD_WITH_SHIFT), (shift, MEAN_SHIFT_STD)])
            if st:
                add("shift_and_spread", st, shift=shift, spread=spread, out=out_share)
                if st == "match":
                    explained |= {"level_shift", "spread_change", "tail_outliers"}
        calm_spread = dist < SPREAD_RATIO
        if calm_spread:
            add("level_shift", _status([(shift, MEAN_SHIFT_STD)]),
                mean_ref=m.get("mean_ref"), mean_cur=m.get("mean_cur"), shift=shift)
        if shift < MEAN_SHIFT_STD:
            add("spread_change", _status([(dist, SPREAD_RATIO)]),
                std_ref=m.get("std_ref"), std_cur=m.get("std_cur"), spread=spread)
    elif m["kind"] == "categorical":
        if m.get("top_cur") is not None and m["top_ref"] <= STUCK_REF_SHARE:
            add("stuck_values", _status([(m["top_cur"], STUCK_SHARE)]),
                value=m["top_label"], share=m["top_cur"])
        if m["new_mass"] >= NEW_VALUE_SHARE:
            add("new_category", "match", values=m["new"], variants=m["variants"])
        if m["vanished_mass"] >= NEW_VALUE_SHARE:
            add("vanished_category", "match", values=m["vanished"])
        if not m["new"] and not m["vanished"] and m.get("mover"):
            mv = m["mover"]
            add("mix_shift", _status([(m["share_delta"], SHARE_SHIFT)]),
                label=mv[0], ref=mv[1], cur=mv[2])
    if "missing_delta" in m:
        add("missing_surge", _status([(m["missing_delta"], MISSING_CHANGE)]),
            ref=m["missing_ref"], cur=m["missing_cur"])
    # a near reading is noise next to a match on the same attribute
    if any(c["match"] == "match" for c in out):
        out = [c for c in out if c["match"] == "match"]
    return out


def read(*, verdict: str, scores: List[Tuple[str, float]], threshold: float,
         summaries: Dict[str, Any], evidence: Optional[Dict[str, Any]] = None,
         profile: Optional[Dict[str, Any]] = None,
         smallest: Optional[Dict[str, int]] = None,
         projection: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Archetype cards for one comparison: dataset-level first, then
    attribute-level grouped by archetype (strongest attribute first)."""
    if verdict == "BLIND":
        return []
    evidence = evidence or {}
    score_of = dict(scores)
    cards: List[Dict[str, Any]] = []

    cols = (profile or {}).get("columns") or {}
    if cols.get("only_ref") or cols.get("only_cur"):
        cards.append({"code": "schema_change", "match": "match",
                      "signal": {"only_ref": cols.get("only_ref") or [], "only_cur": cols.get("only_cur") or []}})
    if smallest:
        n = min(smallest["ref"], smallest["cur"])
        if n < SMALL_SAMPLE / NEAR:
            cards.append({"code": "small_sample", "match": "match" if n < SMALL_SAMPLE else "near",
                          "signal": dict(smallest)})

    # Embeddings (image / text content) separated on the map. Not judged
    # on a handful of items: the map itself carries that caution.
    if isinstance(projection, dict) and projection.get("mixing") is not None \
            and not projection.get("few_points"):
        st = _status([(1 - projection["mixing"], EMBED_SEPARATION)])
        if st:
            cards.append({"code": "embedding_shift", "match": st,
                          "signal": {"mixing": projection["mixing"],
                                     "shift": projection.get("centroid_shift")}})

    n_attrs = len(scores)
    over = [k for k, v in scores if v >= threshold]
    if n_attrs >= 3:  # with two attributes "most of the data" says little
        st = _status([(len(over) / n_attrs, REGIME_SHARE)])
        if st and over:
            cards.append({"code": "regime_shift", "match": st,
                          "signal": {"attrs": over, "n_over": len(over), "n_attrs": n_attrs}})
        elif over and len(over) / n_attrs <= LOCAL_SHARE:
            cards.append({"code": "local_shift", "match": "match",
                          "signal": {"attrs": over, "n_over": len(over), "n_attrs": n_attrs}})

    per_attr: Dict[str, List[Dict[str, Any]]] = {}
    for name, summary in summaries.items():
        if not isinstance(summary, dict) or summary.get("kind") not in ("numeric", "categorical"):
            continue
        for card in _attr_cards(name, _measure(summary, evidence.get(name))):
            per_attr.setdefault(card["code"], []).append(card)
    grouped = []
    for code in ATTR_CODES:
        items = per_attr.get(code) or []
        for status in ("match", "near"):
            same = sorted((c for c in items if c["match"] == status),
                          key=lambda c: score_of.get(c["attr"], 0.0), reverse=True)
            if same:
                grouped.append({"code": code, "match": status,
                                "attrs": [c["attr"] for c in same],
                                "signals": [{"attr": c["attr"], **c["signal"]} for c in same]})
    grouped.sort(key=lambda c: (c["match"] != "match",
                                -max(score_of.get(a, 0.0) for a in c["attrs"])))

    matched_attr = [c for c in grouped if c["match"] == "match"]
    if verdict == "OK" and matched_attr:
        cards.append({"code": "hidden_change", "match": "match",
                      "signal": {"attrs": sorted({a for c in matched_attr for a in c["attrs"]})}})
    if verdict == "OK" and not matched_attr and not any(c["match"] == "match" for c in cards):
        cards.append({"code": "stable", "match": "match",
                      "signal": {"max_score": max(score_of.values()) if score_of else 0.0}})
    return cards + grouped


# ── Write-ups ─────────────────────────────────────────────────────────
#
# Each entry: title, meaning, causes, actions, caveat, and signal (a
# format string filled from the measured numbers; see _signal_fields).

CATALOG: Dict[str, Dict[str, Dict[str, Any]]] = {
    "regime_shift": {
        "en": {"title": "Regime shift",
               "meaning": "Many attributes moved at once. That is more often a change in the world the data comes from (period, market, policy, source) than an error in individual columns.",
               "causes": ["Data from a different period or season", "A change in pricing, product range or channel strategy", "A different source or population being collected"],
               "actions": ["Ask the data owner whether the new state will last (a new normal) or is temporary", "If it lasts, refresh the baseline and consider retraining"],
               "caveat": "Do not fix the columns one by one; a single cause is likely.",
               "signal": "{n_over} of {n_attrs} attributes over the threshold ({attrs})"},
        "ko": {"title": "레짐 전환형",
               "meaning": "여러 속성이 한꺼번에 움직였습니다. 개별 열의 오류보다는 데이터가 나온 환경(기간·시장·정책·출처) 자체가 바뀐 경우에 흔한 모습입니다.",
               "causes": ["기간이나 계절이 다른 데이터", "가격 정책·상품 구성·채널 전략 변경", "데이터 출처나 수집 대상 집단의 변경"],
               "actions": ["새 상태가 계속될지(새 기준) 일시적인지 데이터 담당자와 확인", "계속된다면 기준 데이터를 갱신하고 모델 재학습 검토"],
               "caveat": "열을 하나씩 고치려 하지 마세요. 원인이 하나일 가능성이 큽니다.",
               "signal": "{n_attrs}개 속성 중 {n_over}개가 기준 이상 ({attrs})"},
    },
    "embedding_shift": {
        "en": {"title": "Content shift in embeddings",
               "meaning": "On the embedding map the current items sit apart from the baseline: what the images or texts show or say has changed, not only their measurable properties.",
               "causes": ["New scenes, objects, topics or classes", "A camera, lighting or writing-style change", "A preprocessing change (resizing, colour handling, tokenisation)", "A different data source"],
               "actions": ["Open a few items nearest each centre on the map and compare them by eye", "Check the preprocessing pipeline for recent changes", "Check the class or topic balance of the current data"],
               "caveat": "Surface changes such as lighting also move embeddings; confirm by looking at items before concluding the content changed.",
               "signal": "neighbour mixing {mixing} (1 = evenly mixed) · centre moved {shift}× the baseline spread"},
        "ko": {"title": "임베딩 분리형",
               "meaning": "임베딩 지도에서 현재 항목들이 기준과 떨어져 있습니다. 측정 가능한 속성뿐 아니라 이미지나 문장이 담은 내용 자체가 달라졌다는 뜻입니다.",
               "causes": ["새로운 장면·객체·주제·클래스", "카메라·조명·문체 변화", "전처리 변경 (크기 조정, 색 처리, 토큰화)", "다른 데이터 출처"],
               "actions": ["지도에서 각 중심에 가까운 항목 몇 개를 열어 눈으로 비교", "전처리 과정의 최근 변경 확인", "현재 데이터의 클래스·주제 구성 확인"],
               "caveat": "조명 같은 표면 변화도 임베딩을 움직입니다. 내용이 바뀌었다고 결론 내리기 전에 항목을 직접 보세요.",
               "signal": "이웃 섞임 {mixing} (1 = 고르게 섞임) · 중심 이동 기준 퍼짐의 {shift}배"},
    },
    "local_shift": {
        "en": {"title": "Local shift",
               "meaning": "Most of the data is unchanged; a few attributes moved a lot. Typically one source, sensor or processing step behind those columns changed.",
               "causes": ["A change in the source system, sensor or code table behind the column", "A change in preprocessing logic", "A change confined to one segment"],
               "actions": ["Trace the column's lineage: where it comes from and how it is processed", "Compare the period with deployment or configuration change dates"],
               "caveat": "Do not discard the whole dataset because of the overall verdict.",
               "signal": "only {attrs} over the threshold ({n_over} of {n_attrs})"},
        "ko": {"title": "국소 이동형",
               "meaning": "대부분은 그대로이고 일부 속성만 크게 움직였습니다. 그 열을 만드는 출처·센서·처리 단계 하나가 바뀐 경우가 흔합니다.",
               "causes": ["해당 열의 원천 시스템·센서·코드표 변경", "전처리 로직 수정", "특정 세그먼트에 한정된 변화"],
               "actions": ["해당 열의 계보(어디서 오고 어떻게 가공되는지) 추적", "배포·설정 변경 날짜와 데이터 기간 대조"],
               "caveat": "전체 판정만 보고 데이터 전체를 버리지 마세요.",
               "signal": "{attrs} 만 기준 이상 ({n_attrs}개 중 {n_over}개)"},
    },
    "schema_change": {
        "en": {"title": "Structure change",
               "meaning": "The columns differ from the baseline. A column missing on one side cannot be compared, so it never shows in the verdict.",
               "causes": ["Columns added or removed in the pipeline", "A renamed column", "A different version of the extraction query"],
               "actions": ["Check whether a vanished column was only renamed", "Check that every input column the model uses is present"],
               "actions_added": ["Find out what the new column holds and where it comes from", "Decide whether it should become a model input (it was not inspected here)"],
               "caveat": "Even with an OK verdict, the missing columns were not inspected.",
               "signal": "baseline only: {only_ref} · current only: {only_cur}"},
        "ko": {"title": "구성 변경형",
               "meaning": "열 구성이 기준과 다릅니다. 한쪽에만 있는 열은 비교할 수 없어 판정에 나타나지 않습니다.",
               "causes": ["파이프라인의 열 추가·삭제", "열 이름 변경", "다른 버전의 추출 쿼리"],
               "actions": ["사라진 열이 이름만 바뀐 것인지 확인", "모델이 쓰는 입력 열이 모두 있는지 점검"],
               "actions_added": ["새 열이 무엇을 담고 어디서 오는지 확인", "모델 입력으로 쓸지 결정 (이번 비교에서는 검사하지 않음)"],
               "caveat": "판정이 OK여도 빠진 열은 검사되지 않았습니다.",
               "signal": "기준에만: {only_ref} · 현재에만: {only_cur}"},
    },
    "small_sample": {
        "en": {"title": "Too few samples",
               "meaning": "With this few samples, scores move a lot by chance.",
               "causes": ["A short collection window", "A filter that is too narrow"],
               "actions": ["Collect more data and compare again", "Compare another sample from the same period and see whether the score repeats"],
               "caveat": "Treat the verdict as provisional.",
               "signal": "samples {ref} / {cur}"},
        "ko": {"title": "표본 부족형",
               "meaning": "표본이 적어 점수가 우연에 크게 흔들릴 수 있습니다.",
               "causes": ["짧은 수집 기간", "너무 좁은 필터 조건"],
               "actions": ["표본을 더 모은 뒤 다시 비교", "같은 기간의 다른 표본과 비교해 점수가 재현되는지 확인"],
               "caveat": "지금의 판정은 잠정 판정으로 보세요.",
               "signal": "표본 {ref} / {cur}"},
    },
    "hidden_change": {
        "en": {"title": "Hidden change",
               "meaning": "The verdict is OK, yet some attributes show a change pattern over its threshold. Some changes barely move the distribution score — a mean shift in a column with few distinct values, for example.",
               "causes": ["The score threshold is not sensitive for this column", "A change that is only starting"],
               "actions": ["Track whether the same pattern shows next time", "For an important column, try a lower single-attribute threshold (--attr-threshold)"],
               "caveat": "Do not read OK as \"nothing changed\".",
               "signal": "patterns in {attrs}"},
        "ko": {"title": "잠복 변화형",
               "meaning": "판정은 OK지만 일부 속성에서 기준을 넘는 변화 패턴이 보입니다. 값 종류가 적은 열의 평균 이동처럼, 분포 점수에는 잘 드러나지 않는 변화가 있습니다.",
               "causes": ["점수 기준이 이 열에는 둔감함", "막 시작된 변화"],
               "actions": ["다음 회차에도 같은 패턴이 나오는지 추적", "중요한 열이라면 단일 속성 기준을 낮춰 비교해 보기 (--attr-threshold)"],
               "caveat": "OK를 '변화 없음'으로 읽지 마세요.",
               "signal": "{attrs} 에서 패턴 확인"},
    },
    "stable": {
        "en": {"title": "Stable",
               "meaning": "Neither the verdict criteria nor the change patterns show a notable difference.",
               "causes": ["Data collected under the same conditions", "Differences at the level of sampling noise"],
               "actions": ["Keep the regular inspection schedule"],
               "caveat": "This compares distributions only; a change in the relation to labels (the right answers) needs a separate check.",
               "signal": "highest attribute score {max_score}"},
        "ko": {"title": "안정형",
               "meaning": "판정 기준과 변화 패턴 모두에서 눈에 띄는 차이가 없습니다.",
               "causes": ["같은 조건에서 수집된 데이터", "표본 잡음 수준의 차이"],
               "actions": ["정기 점검 주기 유지"],
               "caveat": "이 비교는 분포만 봅니다. 라벨(정답)과의 관계 변화는 따로 확인해야 합니다.",
               "signal": "최대 속성 점수 {max_score}"},
    },
    "unit_change": {
        "en": {"title": "Unit or scale change",
               "meaning": "The mean and the spread changed by the same factor. That is what a change of unit or scale looks like: same meaning, different numbers.",
               "causes": ["A unit change (KRW → thousand KRW, m → mm, s → ms)", "A change in decimal handling or normalisation upstream"],
               "actions": ["Compare the value of the same item in both periods directly", "Fix the conversion, bring the data to the baseline's unit and compare again"],
               "caveat": "Align the units before retraining anything.",
               "signal": "{attr}: mean {mean_ref} → {mean_cur}, mean ×{ratio} · spread ×{spread}{factor}"},
        "ko": {"title": "단위·스케일 변경형",
               "meaning": "평균과 산포가 같은 배율로 함께 바뀌었습니다. 값의 뜻은 같고 단위나 스케일만 바뀌었을 때 나타나는 모습입니다.",
               "causes": ["단위 변경 (원 → 천 원, m → mm, 초 → 밀리초)", "상위 시스템의 소수점·정규화 처리 변경"],
               "actions": ["두 기간에서 같은 대상의 값을 직접 대조", "변환식을 확정해 기준과 같은 단위로 맞춘 뒤 다시 비교"],
               "caveat": "모델을 다시 학습하기 전에 단위부터 맞추세요.",
               "signal": "{attr}: 평균 {mean_ref} → {mean_cur}, 평균 ×{ratio} · 산포 ×{spread}{factor}"},
    },
    "stuck_values": {
        "en": {"title": "Stuck values",
               "meaning": "The values are nearly fixed to one.",
               "causes": ["A stalled sensor or device", "Missing values filled with a default", "A bug in collection"],
               "actions": ["Check whether the fixed value is a default (0, -1, empty string, …)", "Check the collection logs for the period"],
               "caveat": "A change that looks small may mean the information is gone.",
               "signal": "{attr}: {value} holds {share} of the current data"},
        "ko": {"title": "고착형",
               "meaning": "값이 거의 하나로 고정되었습니다.",
               "causes": ["센서·장비 멈춤", "결측을 기본값으로 채움", "수집 로직 오류"],
               "actions": ["고정된 값이 기본값(0, -1, 빈 문자열 등)인지 확인", "해당 기간의 수집 로그 확인"],
               "caveat": "변화가 작아 보여도 정보가 사라진 상태일 수 있습니다.",
               "signal": "{attr}: 현재 데이터의 {share} 가 {value}"},
    },
    "shift_and_spread": {
        "en": {"title": "Shift with wider spread",
               "meaning": "The values moved to one side and spread out at the same time. Common when equipment degrades or loses calibration, and when a population with different behaviour mixes in.",
               "causes": ["Sensor or equipment ageing, calibration drift", "A different population or source mixing in", "Intermittent spikes or noise"],
               "actions": ["Compare with another device, channel or segment at the same time", "Split the current data by source or segment and see which part moved", "Cut out the most recent window and check the trend"],
               "caveat": "Do not read it only as \"values went up\"; the data may also have become less consistent.",
               "signal": "{attr}: mean moved {shift}σ, spread ×{spread}, outside baseline range {out}"},
        "ko": {"title": "이동·확산형",
               "meaning": "값이 한쪽으로 밀리면서 퍼짐도 커졌습니다. 장비 노후·보정 이탈 같은 열화나, 성격이 다른 집단이 섞여 들어온 경우에 흔한 모습입니다.",
               "causes": ["센서·장비 노후, 보정 이탈", "성격이 다른 집단·출처의 유입", "간헐적 스파이크·잡음"],
               "actions": ["같은 시점의 다른 장비·채널·세그먼트 값과 비교", "현재 데이터를 출처·세그먼트별로 나눠 어느 쪽이 움직였는지 확인", "최근 구간만 잘라 추세 확인"],
               "caveat": "'값이 올랐다'로만 읽지 마세요. 데이터의 일관성 자체가 떨어졌을 수 있습니다.",
               "signal": "{attr}: 평균 {shift}σ 이동, 산포 ×{spread}, 기준 범위 밖 {out}"},
    },
    "level_shift": {
        "en": {"title": "Level shift",
               "meaning": "The distribution kept its shape but moved as a whole.",
               "causes": ["A price or policy change", "A calibration offset", "A different population"],
               "actions": ["Find when the move started and compare with the change history", "If the move is constant, check whether an offset correction is enough"],
               "caveat": "Same shape does not mean small impact: threshold-based rules swing a lot.",
               "signal": "{attr}: mean {mean_ref} → {mean_cur} ({shift}σ)"},
        "ko": {"title": "수준 이동형",
               "meaning": "분포 모양은 비슷한데 전체가 한쪽으로 옮겨갔습니다.",
               "causes": ["가격·정책 변경", "보정 오프셋", "대상 집단 변화"],
               "actions": ["이동이 시작된 시점을 찾아 변경 이력과 대조", "이동폭이 일정하면 오프셋 보정으로 충분한지 검토"],
               "caveat": "모양이 같다고 영향이 작은 것은 아닙니다. 임계값 기반 규칙은 크게 흔들립니다.",
               "signal": "{attr}: 평균 {mean_ref} → {mean_cur} ({shift}σ)"},
    },
    "spread_change": {
        "en": {"title": "Spread change",
               "meaning": "The centre stayed put but the spread changed a lot.",
               "causes": ["More noise, or sources being mixed (wider)", "Clipping or default-value filling (narrower)"],
               "actions": ["Look at the extreme and the repeated values", "Compare the spread per source"],
               "caveat": "Checks that only compare means miss this.",
               "signal": "{attr}: std {std_ref} → {std_cur} (×{spread})"},
        "ko": {"title": "산포 변화형",
               "meaning": "중심은 그대로인데 값의 퍼짐이 크게 달라졌습니다.",
               "causes": ["잡음 증가나 서로 다른 출처의 혼합 (커짐)", "값 잘림(clipping)이나 기본값 채움 (작아짐)"],
               "actions": ["극단값과 반복되는 값 확인", "출처별로 나눠 산포 비교"],
               "caveat": "평균만 비교하는 점검에서는 놓치는 변화입니다.",
               "signal": "{attr}: 표준편차 {std_ref} → {std_cur} (×{spread})"},
    },
    "tail_outliers": {
        "en": {"title": "Outliers in the tails",
               "meaning": "Centre and spread are about the same, but values outside the baseline range appeared.",
               "causes": ["Input errors or spikes", "A new high or low segment"],
               "actions": ["Pull the out-of-range values and check the original records", "Consider a valid-range check"],
               "caveat": "A few extreme values can still move mean-based metrics and training a lot.",
               "signal": "{attr}: outside baseline range {out}"},
        "ko": {"title": "극단값 출현형",
               "meaning": "중심과 퍼짐은 비슷한데 기준 범위 밖의 값이 나타났습니다.",
               "causes": ["입력 오류·스파이크", "새로운 고가·저가 구간 등장"],
               "actions": ["범위 밖 값을 뽑아 원본 기록 확인", "유효 범위 검증 규칙 추가 검토"],
               "caveat": "소수의 극단값도 평균 기반 지표와 학습을 크게 흔들 수 있습니다.",
               "signal": "{attr}: 기준 범위 밖 {out}"},
    },
    "missing_surge": {
        "en": {"title": "Missing values surge",
               "meaning": "Many more values arrive empty.",
               "causes": ["An upstream outage or integration failure", "A field that became optional", "Mismatched join keys"],
               "actions": ["Find the dates and sources where the gaps cluster", "If a step fills gaps with defaults, check its effect"],
               "caveat": "Depending on how gaps are filled, other columns can look shifted too.",
               "signal": "{attr}: missing {ref} → {cur}"},
        "ko": {"title": "결측 증가형",
               "meaning": "빈 값이 크게 늘었습니다.",
               "causes": ["원천 시스템 장애·연동 실패", "선택 입력으로 바뀐 필드", "조인 키 불일치"],
               "actions": ["결측이 몰린 날짜·출처 확인", "결측을 기본값으로 채우는 단계가 있다면 그 영향 점검"],
               "caveat": "결측 처리 방식에 따라 다른 열의 분포까지 바뀌어 보일 수 있습니다.",
               "signal": "{attr}: 결측 {ref} → {cur}"},
    },
    "new_category": {
        "en": {"title": "New categories",
               "meaning": "Values that the baseline never had have arrived.",
               "causes": ["A new product, channel or code", "A spelling variant (case, spaces, typo)", "A missing code mapping"],
               "actions": ["Check whether a new value is another spelling of an existing one", "Check how the model or rules handle a value they have never seen"],
               "caveat": "To a model a new value is an unknown; even a small share can matter.",
               "signal": "{attr}: {values}{variants}"},
        "ko": {"title": "새 범주 유입형",
               "meaning": "기준에 없던 값이 들어왔습니다.",
               "causes": ["새 상품·채널·코드 추가", "표기 변형(대소문자·공백·오타)", "코드 매핑 누락"],
               "actions": ["새 값이 기존 값의 다른 표기인지 확인", "모델·규칙이 처음 보는 값을 어떻게 처리하는지 점검"],
               "caveat": "새 값은 모델에게 '모르는 값'입니다. 비중이 작아도 영향이 클 수 있습니다.",
               "signal": "{attr}: {values}{variants}"},
    },
    "vanished_category": {
        "en": {"title": "Vanished categories",
               "meaning": "Values the baseline had are gone.",
               "causes": ["A discontinued item", "A filter applied upstream", "Codes merged"],
               "actions": ["Check whether the vanished values were merged into another", "Check for changed filters or extraction conditions"],
               "caveat": "The remaining values look bigger by exactly the share that vanished.",
               "signal": "{attr}: {values}"},
        "ko": {"title": "범주 소멸형",
               "meaning": "기준에 있던 값이 사라졌습니다.",
               "causes": ["단종·폐지", "상위 단계의 필터 적용", "코드 통합"],
               "actions": ["사라진 값이 다른 값으로 합쳐졌는지 확인", "필터·추출 조건 변경 여부 확인"],
               "caveat": "사라진 비중만큼 나머지 값의 비중이 커져 보입니다.",
               "signal": "{attr}: {values}"},
    },
    "mix_shift": {
        "en": {"title": "Mix shift",
               "meaning": "The same kinds of values, in different proportions.",
               "causes": ["Seasonal or weekday effects", "A campaign or promotion", "A channel strategy change"],
               "actions": ["Check whether the period difference (season, weekday) explains it", "Compare with the same season a year earlier"],
               "caveat": "A mix change can be real demand; do not call it an error.",
               "signal": "{attr}: {label} {ref} → {cur}"},
        "ko": {"title": "구성비 이동형",
               "meaning": "값의 종류는 같고 비중이 바뀌었습니다.",
               "causes": ["계절·요일 효과", "캠페인·프로모션", "채널 전략 변경"],
               "actions": ["기간 차이(계절·요일)로 설명되는지 확인", "작년 같은 시기와 비교"],
               "caveat": "구성비 변화는 실제 수요 변화일 수 있습니다. 오류로 단정하지 마세요.",
               "signal": "{attr}: {label} {ref} → {cur}"},
    },
}

_FRAGMENTS = {
    "en": {"factor": " (close to ×{factor})", "variants": " — possible spelling variant: {pairs}",
           "none": "—", "period": "Period: {ref} → {cur}"},
    "ko": {"factor": " (×{factor} 에 가까움)", "variants": " — 표기 변형 의심: {pairs}",
           "none": "—", "period": "기간: {ref} → {cur}"},
}


def _n(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return f"{x:.4g}" if isinstance(x, float) else str(x)


def _p(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _signal_fields(code: str, s: Dict[str, Any], lang: str) -> Dict[str, Any]:
    f = _FRAGMENTS[lang]
    out = dict(s)
    if "attrs" in s:
        out["attrs"] = ", ".join(s["attrs"])
    for key in ("only_ref", "only_cur"):
        if key in s:
            out[key] = ", ".join(s[key]) or f["none"]
    if code == "unit_change":
        out["ratio"], out["spread"] = f"{s['ratio']:.3g}", f"{s['spread']:.3g}"
        fac = s.get("factor")
        out["factor"] = f["factor"].format(factor=f"{fac:g}" if fac >= 1 else f"1/{1 / fac:g}") if fac else ""
    for key in ("mean_ref", "mean_cur", "std_ref", "std_cur", "value"):
        if key in s and not isinstance(s[key], str):
            out[key] = _n(s[key])
    if "shift" in s:
        out["shift"] = _n(round(s["shift"], 2)) if not math.isinf(s["shift"]) else "∞"
    if "spread" in s and code != "unit_change":
        out["spread"] = "—" if s["spread"] is None else f"{s['spread']:.2f}"
    for key in ("out", "share", "max_score"):
        if key in s:
            out[key] = f"{s[key]:.3f}" if key == "max_score" else _p(s[key])
    if code in ("missing_surge", "mix_shift"):
        out["ref"], out["cur"] = _p(s["ref"]), _p(s["cur"])
    if code == "small_sample":
        out["ref"], out["cur"] = f"{s['ref']:,}", f"{s['cur']:,}"
    if code == "embedding_shift":
        out["mixing"] = f"{s['mixing']:.2f}"
        out["shift"] = "—" if s.get("shift") is None else f"{s['shift']:.2f}"
    if "values" in s:
        out["values"] = ", ".join(f"{k} ({_p(v)})" for k, v in s["values"])
    if code == "new_category":
        pairs = ", ".join(f"{a} ≈ {b}" for a, b in s.get("variants") or [])
        out["variants"] = f["variants"].format(pairs=pairs) if pairs else ""
    return out


# ── Criteria ──────────────────────────────────────────────────────────
#
# When each archetype is read, in words. Placeholders are filled from the
# thresholds above, so the report cards and the generated docs follow the
# code when a threshold changes.

WHEN: Dict[str, Dict[str, str]] = {
    "schema_change": {"en": "at least one column exists on one side only",
                      "ko": "한쪽에만 있는 열이 하나 이상"},
    "small_sample": {"en": "fewer than {small} samples on either side",
                     "ko": "어느 쪽이든 표본이 {small}개 미만"},
    "embedding_shift": {"en": "neighbour mixing on the embedding map at or below {mixing} (only with {few} or more items per side)",
                        "ko": "임베딩 지도의 이웃 섞임이 {mixing} 이하 (한쪽 항목이 {few}개 이상일 때만)"},
    "regime_shift": {"en": "3 or more attributes, of which {regime} or more score at or over the threshold",
                     "ko": "속성 3개 이상 중 {regime} 이상이 기준 점수 이상"},
    "local_shift": {"en": "3 or more attributes, of which at most {local} score at or over the threshold",
                    "ko": "속성 3개 이상 중 {local} 이하만 기준 점수 이상"},
    "hidden_change": {"en": "verdict OK while at least one attribute pattern matches",
                      "ko": "판정은 OK인데 속성 유형이 하나 이상 해당"},
    "stable": {"en": "verdict OK and no pattern matches",
               "ko": "판정이 OK이고 해당하는 유형이 없음"},
    "unit_change": {"en": "mean changed by ×{unit} or more (or ÷{unit}) and the spread by about the same factor",
                    "ko": "평균이 {unit}배 이상(또는 1/{unit} 이하) 바뀌고 산포도 거의 같은 배율로 바뀜"},
    "stuck_values": {"en": "current values nearly one (numeric: spread 0; categorical: one value {stuck} or more, where the baseline's top value held {stuck_ref} or less)",
                     "ko": "현재 값이 거의 하나 (수치형은 산포 0, 범주형은 한 값이 {stuck} 이상 — 기준에서는 최빈값이 {stuck_ref} 이하)"},
    "shift_and_spread": {"en": "mean moved {shift} or more and the spread grew ×{spread_shift} or more",
                         "ko": "평균이 {shift} 이상 이동하고 산포도 {spread_shift}배 이상 커짐"},
    "level_shift": {"en": "mean moved {shift} or more; spread within ×{spread}",
                    "ko": "평균이 {shift} 이상 이동, 산포는 {spread}배 미만으로 비슷"},
    "spread_change": {"en": "mean moved less than {shift}; spread changed ×{spread} or more (or ÷{spread})",
                      "ko": "평균 이동은 {shift} 미만, 산포가 {spread}배 이상(또는 1/{spread} 이하) 달라짐"},
    "tail_outliers": {"en": "median moved less than {shift} and the interquartile width stayed within ×{bulk}, yet {out} or more of the values lie outside the baseline range",
                      "ko": "중앙값 이동 {shift} 미만, 사분위 범위 ×{bulk} 이내인데 기준 범위 밖 값이 {out} 이상"},
    "missing_surge": {"en": "missing rate up {missing} or more",
                      "ko": "결측률이 {missing} 이상 늘어남"},
    "new_category": {"en": "values absent from the baseline make up {new} or more",
                     "ko": "기준에 없던 값이 {new} 이상"},
    "vanished_category": {"en": "baseline values holding {new} or more are gone",
                          "ko": "기준에 있던 값({new} 이상)이 사라짐"},
    "mix_shift": {"en": "no new or vanished values; one value's share moved {share} or more",
                  "ko": "새 값·소멸 없이 한 값의 비중이 {share} 이상 바뀜"},
}


def limits() -> Dict[str, Any]:
    """The thresholds, formatted for the criteria sentences."""
    from .embedding import FEW_POINTS

    return {"small": SMALL_SAMPLE, "regime": f"{REGIME_SHARE * 100:.0f}%", "local": f"{LOCAL_SHARE * 100:.0f}%",
            "unit": f"{UNIT_RATIO:g}", "stuck": f"{STUCK_SHARE * 100:.0f}%", "stuck_ref": f"{STUCK_REF_SHARE * 100:.0f}%",
            "shift": f"{MEAN_SHIFT_STD:g}σ", "spread": f"{SPREAD_RATIO:g}", "spread_shift": f"{SPREAD_WITH_SHIFT:g}",
            "bulk": f"{BULK_STEADY:g}", "out": f"{OUT_OF_RANGE * 100:.0f}%", "missing": f"{MISSING_CHANGE * 100:.0f}%p",
            "new": f"{NEW_VALUE_SHARE * 100:.0f}%", "share": f"{SHARE_SHIFT * 100:.0f}%p",
            "mixing": f"{1 - EMBED_SEPARATION:g}", "few": FEW_POINTS, "near": f"{NEAR * 100:.0f}%"}


def criterion_text(code: str, lang: str) -> str:
    entry = WHEN[code]
    return (entry.get(lang) or entry["en"]).format(**limits())


def card_text(card: Dict[str, Any], lang: str, *,
              profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Display strings for one archetype card."""
    entry = CATALOG[card["code"]].get(lang) or CATALOG[card["code"]]["en"]
    signals = card.get("signals") or [card.get("signal") or {}]
    lines = [entry["signal"].format(**_signal_fields(card["code"], s, lang)) for s in signals]
    if card["code"] in ("regime_shift", "mix_shift") and profile:
        ref_p, cur_p = (profile.get("ref") or {}).get("period"), (profile.get("cur") or {}).get("period")
        if ref_p and cur_p:
            lines.append(_FRAGMENTS[lang]["period"].format(
                ref=f"{ref_p['start']} ~ {ref_p['end']}", cur=f"{cur_p['start']} ~ {cur_p['end']}"))
    actions = entry["actions"]
    if card["code"] == "schema_change" and not (card.get("signal") or {}).get("only_ref"):
        actions = entry["actions_added"]  # columns were only added
    return {"title": entry["title"], "meaning": entry["meaning"], "causes": entry["causes"],
            "criterion": criterion_text(card["code"], lang), "actions": actions, "caveat": entry["caveat"], "signals": lines}
