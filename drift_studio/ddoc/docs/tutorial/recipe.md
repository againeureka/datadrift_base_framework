# `ddoc recipe` — multi-step workflows in one YAML

Round-16 — chain ddoc 의 여러 명령어를 YAML 한 파일로 묶어 한 번에
실행. CI / 스케줄 작업 / "이 4 단계 항상 같이 돌림" 워크플로의 단일
입력. gpu-tunnel `recipes/` 패턴 참조.

## 빠른 시작

샘플 recipe 가 패키지에 동봉됨:

```bash
ddoc recipe validate recipes/timeseries_smoke.yaml
ddoc recipe run recipes/timeseries_smoke.yaml
```

또는 dry-run 으로 각 step 의 argv 만 미리 보기:

```bash
ddoc recipe run recipes/timeseries_smoke.yaml --dry-run
```

## 스키마

```yaml
name: <human label, optional>
description: <multi-line, optional>
workspace: <auto-envelope 저장 dir, optional — 기본은 recipe 옆에
            .ddoc-recipe-out/<recipe stem>/>
vars:
  <key>: <value>            # ${vars.<key>} 로 step 안에서 참조
steps:
  - id: <unique per recipe>
    run: <kind>             # 아래 표 참조
    with:                   # ddoc CLI 옵션을 1대1 매핑
      <option>: <value>
    out: <optional explicit output path>
```

### 지원 step kind

| `run` | ddoc CLI 매핑 | 주요 `with` 키 |
|---|---|---|
| `fetch` | `ddoc fetch` | `dest`, `symlink`, `config` |
| `examples.generate` | `ddoc examples generate` | `out`, `scenario` |
| `analyze.eda` | `ddoc analyze eda` | `data_path`, `quiet` 등 |
| `analyze.drift` | `ddoc analyze drift` | `data_path_ref/cur`, `detector`, `fusion`, `quiet` |
| `report.render` | `ddoc report render` | `input`, `out`, `format`, `title` |
| `export.drift_report` | `ddoc export drift-report` | `target`, `config` |

`ddoc recipe kinds` 로 항상 최신 목록 확인 가능.

### 참조 (substitution) 문법

`with` 의 값과 `out` 필드 안에서 다음 placeholder 가 인식됨:

| 형태 | 의미 |
|---|---|
| `${vars.<name>}` | recipe 의 `vars` 섹션 |
| `${env.<NAME>}` | 프로세스 환경변수 |
| `${steps.<id>.output}` | 이전 step 의 산출 path (자동 추론) |
| `${steps.<id>.json}` | 이전 step 의 JSON envelope (객체 그대로) |
| `${steps.<id>.json.<dotted>}` | envelope 안의 dotted path 값 |

전체 문자열이 단일 placeholder 면 타입 보존 (envelope dict 가 그대로
넘어감). 부분 치환은 문자열로 변환.

### Auto envelope persistence

`analyze.drift` 와 `analyze.eda` 같이 stdout 으로 envelope 만 내보내는
step 들은 자동으로 `<workspace>/<step_id>.envelope.json` 에 atomic
저장됨. 따라서 후속 step 이 `${steps.drift.output}` 만으로 결과 파일
경로를 받을 수 있음 (별도 "save to file" step 불필요).

명시적 `out:` 을 step 에 두면 그 경로가 우선.

## 실행 contract

* 첫 실패 step 에서 정지 — error envelope 을 결과의 `error` 필드로
  bubble.
* `--json` 출력은 `{status, recipe, steps: [...]}` 형태로 모든 step
  결과 포함.
* `--dry-run` 은 substitution 까지 다 해소한 argv 만 출력하고 실제
  ddoc 호출은 건너뜀 — recipe 디버깅에 유용.

## YAML 작성 팁

`with` 안에서 `${...}` 를 쓸 때는 인라인 mapping (`{ key: value }`)
대신 block mapping 으로 풀어 쓰는 편이 안전:

```yaml
# ❌ flow-mapping 안의 ${} 는 YAML parser 가 nested map 으로 오해
with: { out: ${vars.x}, scenario: shifted }

# ✅ block mapping 또는 명시적 quote
with:
  out: ${vars.x}
  scenario: shifted

# ✅ 또는
with: { out: "${vars.x}", scenario: shifted }
```

## REST API 와의 관계

레시피는 CLI 우선 — `ddoc recipe run` 이 메인 path. `ddoc serve` 의
신규 GUI 안에서 한 번에 실행하는 multi-step 통합은 Round-17+ 후속.
이미 GUI 의 result viz 패널은 `{status, recipe, steps: [...]}` 형태의
envelope 을 카드 + 단계 목록으로 표시.

## REST endpoints (Round 17)

`ddoc serve` 띄우면 동일한 동작이 HTTP 로 노출됨:

| Method | Path | 설명 |
|---|---|---|
| POST | `/recipe/validate` | 본문 `{yaml: "..."}` 또는 `{path: "..."}`. 파싱 + 구조 검증만. |
| POST | `/recipe/run` | 동기 실행. `dry_run: true` 도 지원. |
| POST | `/recipe/run/stream` | SSE: `event: progress` per step + `event: result` 마지막. |

```bash
# 인라인 YAML 검증
curl -s -X POST http://localhost:8765/recipe/validate \
  -H 'Content-Type: application/json' \
  -d '{"yaml": "name: tiny\nsteps:\n  - id: a\n    run: examples.generate\n    with: { modality: timeseries, out: /tmp/x, scenario: shifted }"}'
# → {"status":"ok","recipe":"tiny","step_count":1,"issues":[]}

# 실제 실행 (서버 측 path)
curl -s -X POST http://localhost:8765/recipe/run \
  -H 'Content-Type: application/json' \
  -d '{"path": "recipes/timeseries_smoke.yaml"}'

# 스트리밍 진행 표시
curl -N -X POST http://localhost:8765/recipe/run/stream \
  -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
  -d '{"path": "recipes/timeseries_smoke.yaml"}'
```

## Conditional step — `when:` (Round 17)

각 step 에 optional `when:` 표현식을 붙여 조건부 실행:

```yaml
steps:
  - id: drift
    run: analyze.drift
    with: { ... }
  - id: heavy_report
    run: report.render
    when: "${steps.drift.json.overall_score} > 0.25"
    with: { ... }
```

조건이 거짓이면 step 의 결과에 `skipped: true, skipped_reason: "when"`
이 표시됨. 평가기는 안전한 mini-eval — 리터럴 (number / string / bool /
null), 비교 (`==`, `!=`, `<`, `<=`, `>`, `>=`), 불 연산 (`and`, `or`,
`not`), 단항 (+/-), 괄호만 허용. `Name` / `Call` / `Attribute` 같은
임의 평가는 거부 (보안).

## GUI Recipe tab (Round 17)

`ddoc serve` 의 7번째 탭 `Recipe` 에서 :
* YAML 텍스트 편집 (default 로 샘플 시나리오 포함)
* 또는 server-side path 모드
* `dry-run`, `validate only`, `Use streaming` 토글
* SSE 모드 시 step 별 chip (`gen_pair (examples.generate) — ok`,
  `report_md (report.render) — skip·when`) 누적
* 결과 패널이 recipe envelope 을 카드로 시각화 (Round 16 의 viz 와
  연결)

## Parallel block — `parallel:` (Round 18)

독립 step 들을 동시 실행:

```yaml
steps:
  - id: pre
    run: fetch
    with: { source_uri: file:///mnt/share/audit, dest: /tmp/audit }

  - parallel:
      - id: drift_a
        run: analyze.drift
        with: { data_path_ref: /tmp/audit/site_a/ref, data_path_cur: /tmp/audit/site_a/cur, quiet: true }
      - id: drift_b
        run: analyze.drift
        with: { data_path_ref: /tmp/audit/site_b/ref, data_path_cur: /tmp/audit/site_b/cur, quiet: true }

  - id: report
    run: report.render
    with: { input: "${steps.drift_a.output}", out: /tmp/audit/report.html, format: html }
```

* parallel block 의 자식 step 들은 *동일 ctx 스냅샷* (block 직전 상태)
  을 공유 — 형제 step 끼리는 서로 참조 불가.
* block 종료 시 모든 자식의 결과가 ctx 에 merge → 후속 직렬 step 이
  자유롭게 `${steps.<child_id>.json/output}` 참조.
* 어느 자식이 실패해도 나머지는 끝까지 실행 (디버깅 친화). 첫 실패의
  envelope 이 결과의 `error` 필드로 bubble.
* `dry_run` 도 parallel 그대로 처리 (substitution 만 해소된 argv 출력).

## 후속 (Round 19+)

* recipe 라이브러리 (사이트 별 공유 + 버전 관리)
* `with` 안에서 ${...} 외 다른 expression (지금은 substitution + 비교만)
