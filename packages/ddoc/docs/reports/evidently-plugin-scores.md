# evidently 플러그인의 열 점수 반전 — 확인 보고

2026-11-27 · 대상: `ddoc-plugin-evidently` 0.1.0 · 수정: 작업 브랜치 `restructure/ddoc-top-level` 의 커밋 `1df4833`
(원 저장소 `main` 반영 전)

## 요약

evidently 플러그인은 Evidently 가 열마다 낸 값을 모두 p값으로 읽어 `1 − 값` 을 열 점수로 냅니다. 그런데 Evidently 는
데이터가 1,000행을 넘으면 p값이 아니라 **거리**(수치 열은 정규화 Wasserstein, 범주 열은 Jensen–Shannon)를
냅니다(두 쪽 행 수가 같을 때 확인). 그 결과 크게 바뀐 열은 0점에 drifted=False, 바뀌지 않은 열은 1점 가까이에 drifted=True 가 됩니다. 1,000행
이하에서는 p값이라 순서는 맞지만, `1 − p` 는 변화의 크기가 아니어서 바뀌지 않은 열도 대개 0.25 를 넘습니다.

원 저장소 `main` 의 `ddoc diff --detector evidently:default` 는 이 점수로 판정합니다. 같은 분포에서 뽑은 데이터 쌍
60개 중 59개를 DRIFT 로 판정했고, "가장 변한 속성"에 바뀌지 않은 열을 올렸습니다. `ddoc analyze drift` 도 같은
점수를 그대로 보여 줍니다. 반면 Evidently 가 직접 낸 드리프트 열 비율(`overall_score`)은 맞습니다.

## 영향 범위

**코드.** `plugins/ddoc-plugin-evidently/ddoc_plugin_evidently/plugin.py` 의 `_run_evidently` (원 저장소 `main` 의
235–239행):

```python
score = max(0.0, min(1.0, 1.0 - float(p_value)))
...
"p_value": float(p_value),
"drifted": float(p_value) < 0.05,
```

**공개 이력** (원 저장소 `keti-datadrift/datadrift_base_framework`, 태그 없음):

| 날짜 | 커밋 | 경로 | 상태 |
|---|---|---|---|
| 2026-05-25 | `59a25dc` | `drift_studio/ddoc/plugins/ddoc-plugin-evidently/` | 처음 추가, 같은 코드 |
| 2026-10-15 | `d4af0af` | (위 경로) | 삭제 |
| 2026-11-18 | `3876a52` (PR #139, 병합 `64592bb`) | `plugins/ddoc-plugin-evidently/` | 다시 추가, 같은 코드. **현재 `main`** |

플러그인은 PyPI 에 올라가 있지 않습니다(확인 시점 `pypi.org/pypi/ddoc-plugin-evidently` 404). 소스에서 설치한
경우만 해당합니다.

**조건.** Evidently 0.7 이상(확인한 판: 0.7.23, pandas 2.3.3, Python 3.11). `ddoc-plugin-evidently` 는 detector 가
`evidently:` 로 시작할 때만 동작하므로, `--detector evidently:*` 를 지정한 실행만 해당합니다.

| 명령 (원 저장소 `main`) | 영향 |
|---|---|
| `ddoc diff --detector evidently:*` | 판정, 점수, "가장 변한 속성" |
| `ddoc analyze drift --detector evidently:*` | `attribute_drifts`, `details` 의 `drifted` |
| 같은 detector 를 쓰는 레시피 단계 | 같은 플러그인 출력을 쓰므로 위와 같을 것 (실행해 보지는 않음) |

## 무엇이 잘못되었나

Evidently 의 `DataDriftPreset` 은 열마다 검정을 고르고, `ValueDrift` 지표의 값은 그 검정에 따라 p값이거나
거리입니다. 지표 이름에 검정과 기준이 들어 있습니다(예: `ValueDrift(column=price,method=Wasserstein distance
(normed),threshold=0.1)`). 확인한 선택은 다음과 같습니다.

| 열 | 쪽마다 1,000행 이하 | 1,001행 이상 |
|---|---|---|
| 수치 | K-S p_value | Wasserstein distance (normed), 기준 0.1 |
| 범주 (값 2개) | Z-test p_value | Jensen-Shannon distance, 기준 0.1 |
| 범주 (값 3개, 600행) | chi-square p_value | (확인 안 함) |

p값은 작을수록, 거리는 클수록 드리프트입니다. 플러그인은 둘을 구분하지 않고 모든 값에 `1 − 값` 과
`값 < 0.05` 를 적용했습니다. 이름도 `p_value` 로 남겨, 결과만 보고는 거리인 줄 알 수 없습니다.

## 재현

원 저장소 `main`(`64592bb`)의 `ddoc/` 와 `plugins/ddoc-plugin-evidently/` 를 설치한(또는 `PYTHONPATH` 에 더한)
환경에서:

```python
import random
from ddoc_plugin_evidently.plugin import EvidentlyDriftPlugin

rng = random.Random(0)
n = 2000                                          # 1,000 이하로 바꾸면 반전이 사라짐
base = [rng.gauss(0, 1) for _ in range(n)]
for label, cur in (("3σ 이동", [rng.gauss(3, 1) for _ in range(n)]),
                   ("변화 없음", [rng.gauss(0, 1) for _ in range(n)])):
    out = EvidentlyDriftPlugin().drift_detect(
        "ref", "cur", "", "", "", "", detector="evidently:default",
        cfg={"baseline_numerical": {"x": base}, "current_numerical": {"x": cur}}, output_path="out")
    print(label, out["attribute_drifts"]["x"], out["details"]["x"])
```

수치 열 하나, 현재 데이터를 3σ 이동했을 때와 바꾸지 않았을 때:

| 쪽마다 행 | Evidently 검정 | 3σ 이동: 점수 · drifted · 값 | 변화 없음: 점수 · drifted · 값 |
|---|---|---|---|
| 500 | K-S p_value | 1.000 · True · 5.8e-205 | 0.040 · False · 0.960 |
| 1,000 | K-S p_value | 1.000 · True · 0 | 0.086 · False · 0.914 |
| 1,001 | Wasserstein (normed) | **0.000 · False** · 3.074 | **0.955 · True** · 0.045 |
| 2,000 | Wasserstein (normed) | **0.000 · False** · 3.068 | **0.964 · True** · 0.036 |

범주 열 하나(값 2개, 70/30 → 30/70):

| 쪽마다 행 | Evidently 검정 | 바뀐 쪽: 점수 · drifted | 변화 없음: 점수 · drifted |
|---|---|---|---|
| 500 | Z-test p_value | 1.000 · True | 0.000 · False |
| 1,000 | Z-test p_value | 1.000 · True | — |
| 1,001 | Jensen-Shannon distance | **0.713 · False** | — |
| 2,000 | Jensen-Shannon distance | **0.713 · False** | **1.000 · True** |

## 판정에 미친 영향

원 저장소 `main` 의 `ddoc diff`, 내장 비교기(기본)와 `--detector evidently:default` 비교. 데이터는 열 셋
(price 수치, qty 1–5 정수, channel 범주)이고, 레짐 전환은 Drift Zoo 의 `regime_shift` 입니다.

| 데이터 | 쪽마다 행 | 내장 비교기 | evidently |
|---|---|---|---|
| 변화 없음 | 2,000 | OK | **DRIFT** — channel 0.992 · qty 0.970 · price 0.963 |
| price 만 3σ 이동 | 2,000 | DRIFT — price 0.736 | DRIFT — channel 0.979 · qty 0.956 · **price 0.000** |
| price 만 3σ 이동 | 500 | DRIFT — price 0.789 | DRIFT — price 1.000 · qty 0.864 · channel 0.698 |
| 레짐 전환 | 2,328 | DRIFT — price 1.000 | DRIFT — qty 0.990 · day 0.966 · channel 0.590 · **price 0.000** |

같은 분포에서 따로 뽑은 쌍(위와 같은 세 열) 20개씩:

| 쪽마다 행 | 내장 비교기 DRIFT | evidently DRIFT |
|---|---|---|
| 300 | 0/20 | 19/20 |
| 1,000 | 0/20 | 20/20 |
| 2,000 | 0/20 | 20/20 |

1,000행 이하의 오경보는 거리 반전이 아니라 `1 − p` 를 크기로 쓴 탓입니다. 변화가 없으면 연속 검정의 p 는 0과 1
사이에 고르게 퍼지므로, 열 하나가 단일 속성 기준 0.25 를 넘을 확률이 약 75% 이고 열이 셋이면 거의 늘 하나는
넘습니다.

`ddoc analyze drift --detector evidently:default` (price 만 3σ 이동, 2,000행)의 출력도 같습니다:
`attribute_drifts` = price 0.0 · qty 0.956 · channel 0.979, `drifted` = price False · qty True · channel True.

## 함께 확인한 것

1. **플러그인의 `drifted` 가 Evidently 자신의 판단과 어긋납니다.** 패키지의 기상 기록 예제(`yearly-reset`, 17열)에서
   Evidently 의 드리프트 열 비율은 16/17(0.941)인데, 플러그인의 `drifted` 는 0/17 입니다.
2. **시각 열을 범주 열로 넘깁니다.** CSV 를 읽을 때 문자열 열을 모두 범주로 분류해, 시각 열도 Evidently 가
   Jensen–Shannon 거리로 비교합니다(위 예제에서 0.83). 두 기간의 시각은 늘 달라 이 열은 항상 드리프트로 셉니다.
3. **`evidently:chi2` · `evidently:wasserstein` · `evidently:psi` 가 검정을 고르지 않습니다.** 어떤 키든
   `DataDriftPreset()` 의 기본 선택으로 돌고, 키는 결과의 `method` 에만 남습니다(코드 주석의 "PoC").

## 수정 (`1df4833`)

- **플러그인.** 지표 이름의 검정으로 p값과 거리를 가립니다(이름에 `p_value` 가 있거나 정확 검정이면 p값).
  `details` 는 열마다 `value`, `kind`(`p_value` | `distance`), `test`, `threshold`, `drifted` 를 남기고, `drifted` 는
  p값이면 `값 < 기준`, 거리면 `값 ≥ 기준` 입니다. 열 점수는 p값이면 `1 − p`, 거리면 그 거리(최대 1)입니다. CSV
  경로로 비교했을 때는 결과에 `attribute_source: "csv_columns"` 를 남깁니다.
- **`ddoc diff`.** 결과에 `attribute_source: "csv_columns"` 가 있으면 같은 CSV 를 내장 비교기와 똑같이(시간 열·누적
  열 제외, 하루 단위 잡음 검정까지) 매겨 판정합니다. Evidently 의 수치는 `plugin_scores` 와 `details` 에 남습니다.
- **확인.** 위의 같은 분포 쌍에서 evidently DRIFT 0/20(300·1,000·2,000행). 위 표의 데이터는 변화 없음 OK,
  price 3σ 이동 DRIFT(price 0.858, 500행은 0.888), 레짐 전환 DRIFT(price 1.000 · channel 0.493)로 내장 비교기와
  같습니다. `yearly-reset` 에서 플러그인의 `drifted` 는 16/17 로 Evidently 의 비율과 맞습니다.
- **테스트.** 플러그인: `test_large_numeric_samples_are_judged_by_a_distance`,
  `test_csv_files_are_named_as_the_source`, `test_p_value_tests_are_told_from_distances`. ddoc:
  `test_diff_with_evidently_is_judged_on_ddoc_scale`.
- **호환성.** `details` 의 `p_value` 키가 `value` 와 `kind` 로 바뀝니다. 이 키를 읽는 스크립트는 고쳐야 합니다.

## 남은 것

- 원 저장소 `main` 에 수정이 반영되지 않았습니다.
- 위 "함께 확인한 것"의 2(시각 열)와 3(검정 선택 키)은 고치지 않았습니다. 2는 `ddoc diff` 의 판정에는 영향이
  없고(내장 비교기가 시각 열을 뺌), Evidently 자신의 비율(`plugin_scores.overall_score`)과 `details` 에 남습니다.
- p값 검정의 열 점수 `1 − p` 는 여전히 변화의 크기가 아닙니다. `ddoc diff` 는 이제 이 점수로 판정하지 않지만,
  `ddoc analyze drift` 와 `plugin_scores` 에는 그대로 나옵니다.
- 원 저장소 `main` 에서 `--detector evidently:*` 로 얻은 판정·열 점수는 다시 확인해야 합니다. 기준 데이터가
  1,000행을 넘었다면 열 순서가 뒤집혀 있고, 그 이하라도 바뀌지 않은 열이 높게 나왔을 수 있습니다.
