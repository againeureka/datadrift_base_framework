# 데이터 분석

`ddoc diff` 가 한 줄 판정이라면, `ddoc analyze` 는 플러그인 결과를 그대로
보여주는 저수준 명령입니다. 검출기 선택, 여러 modality 결과 합치기,
진행 상황 스트리밍이 필요할 때 씁니다.

## Drift 감지 — `ddoc analyze drift`

```bash
ddoc analyze drift --data-path-ref data/ref --data-path-cur data/cur
ddoc analyze drift --data-path-ref data/ref --data-path-cur data/cur --json --quiet
```

| 옵션 | 설명 |
|---|---|
| `--data-path-ref` / `--data-path-cur` | 기준 / 현재 데이터 경로 (둘 다 필수) |
| `--detector` | 검출 전략. `ddoc plugin detectors` 로 목록 확인 ([cookbook](../tutorial/detectors.md)) |
| `--with-embeddings` | 플러그인이 임베딩을 즉석 계산 (vision / text, CLIP 필요) |
| `--fusion` / `--fusion-weights` | 여러 modality 점수를 하나로 합침 (`none` / `weighted` / `max` / `joint`) |
| `--json` | 결과 envelope 를 stdout 에 JSON 한 객체로 |
| `--ndjson-progress` | 진행 상황을 stderr 에 NDJSON 으로 (오케스트레이터용) |
| `--quiet` | 플러그인 출력 숨김 |

두 경로 중 하나라도 빠지면 `path_mode_incomplete` 오류와 함께 exit code 2
로 끝납니다.

## EDA — `ddoc analyze eda`

```bash
ddoc analyze eda --data-path data/cur
ddoc analyze eda --data-path data/cur --json --quiet
```

| 옵션 | 설명 |
|---|---|
| `--data-path` | 분석할 데이터 경로 (필수) |
| `--invalidate-cache` | 기존 분석 캐시를 무시하고 다시 분석 |
| `--sample-rows N` | 내장 CSV 프로필에 앞 N행 포함 (기본 끔) |
| `--json` / `--ndjson-progress` / `--quiet` | drift 와 동일 |

이미지·텍스트·오디오·시계열은 `eda_run` hook 을 구현한 플러그인이 분석합니다. 플러그인이
처리하지 않는 CSV 는 내장 프로파일러가 데이터 하나를 기술합니다.

- 규모·기간(날짜·시간 열)·중복 행 비율, 열별 유형·결측률·고유값·분포
- 품질 소견: 아래 표의 판단 기준에 맞는 열·데이터 (보고서 카드에도 같은 문장이 나옵니다)

<!-- generated:profile-findings — scripts/docs.py build -->
현재 10가지입니다.

| 소견 | 범위 | 판단 기준 |
|---|---|---|
| 행 수 부족 | 데이터 전체 | 행이 30개 미만 |
| 중복 행 | 데이터 전체 | 앞의 행과 완전히 같은 행이 1% 이상 |
| 빈 열 | 열 | 값이 하나도 없음 |
| 상수 열 | 열 | 모든 값이 같음 |
| 거의 상수인 열 | 열 | 한 값이 95% 이상 |
| 결측 많음 | 열 | 결측 20% 이상 |
| 식별자형 열 | 열 | 값 50개 이상 중 95% 이상이 서로 다름 (정수 열은 값의 범위도 값 개수와 비슷할 때) |
| 먼 극단값 | 열 | 사분위 ± 3 IQR 밖 값이 1% 이상 |
| 표기 변형 | 열 | 대소문자·공백·구분자만 다른 값이 있음 |
| 텍스트로 저장된 숫자 | 열 | 텍스트 열인데 값의 90% 이상이 숫자로 읽힘 |
<!-- /generated:profile-findings -->

```bash
ddoc analyze eda --data-path aug.csv --json --quiet > eda.json
ddoc report render -i eda.json -o profile.html --lang ko   # 데이터 프로필 보고서
```

보고서는 진단서와 같은 형식(머리글, 데이터 도면, 소견 카드, 화면 테마, 인쇄)이며, 극단값이
있으면 분포 그림은 몸통 범위로 그리고 뺀 비중을 함께 적습니다.

## 결과를 리포트로

```bash
ddoc analyze drift --data-path-ref data/ref --data-path-cur data/cur --json --quiet > drift.json
ddoc report render -i drift.json -o drift.html
ddoc report render -i drift.json -o drift.md --format md
```

drift envelope 은 `ddoc diff` 와 같은 기준(0.15 / 0.25)으로 판정된 뒤 같은 진단서로
렌더됩니다. 기준은 `--threshold` / `--attr-threshold` 로 바꿀 수 있습니다.

## 다음 단계

- [detector cookbook](../tutorial/detectors.md)
- [레시피](../tutorial/recipe.md) — analyze → report → export 를 한 파일로
- [명령어 레퍼런스](../reference/commands.md)
