# 명령어 레퍼런스

모든 명령은 `--help` 로 최신 옵션을 확인할 수 있습니다.
`ddoc showcmd` 는 등록된 명령 트리를 보여줍니다.

리포트와 안내 문구의 언어는 `--lang` 또는 환경변수 `DDOC_LANG`
(`en` · `ko`) 으로 고릅니다. `--json` 출력의 키와 판정 값(`OK`, `DRIFT`,
`high` 등)은 언어와 관계없이 영어로 유지됩니다.

## `ddoc diff <ref> <cur>`

두 데이터셋의 한 줄 drift 판정. exit code: `0` OK · `1` DRIFT · `2` BLIND.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--threshold` | 0.15 | 전체 점수 DRIFT 임계값 |
| `--attr-threshold` | 0.25 | 속성 하나만으로 DRIFT 가 되는 임계값 |
| `--detector` | `default` | 플러그인에 넘길 검출기 (`ddoc plugin detectors`) |
| `--json` | | 판정 JSON 한 객체를 stdout 에 |
| `--verbose` | | 플러그인 진단 메시지 표시 |
| `--report <path>` | | 진단서 저장. 확장자로 형식 결정: `.html`(단일 파일 진단서), `.md`(PR·이슈용 Markdown), `.pdf` |
| `--lang` | `$DDOC_LANG`, 없으면 `en` | 진단서·안내 문구·터미널 출력 언어 (`en` · `ko`) |
| `--issuer` | `$DDOC_ISSUER`, 없으면 `ddoc` | 진단서 머리글에 들어갈 발행처 |
| `--max-rows N` | `200000` | CSV 한쪽이 이보다 크면 균등 무작위 표본 N행으로 비교 (`0` = 전체). 진단서에 "전체 중 N행 무작위 추출"로 표시 |
| `--with-embeddings` | 끔 | 이미지·텍스트 플러그인이 임베딩까지 비교 (모델 로드, 느림). 진단서에 임베딩 지도 추가 |
| `--sample-rows N` | `0` (끔) | CSV 양쪽의 앞 N행(최대 20)을 진단서와 JSON 에 포함. 원본 데이터가 담기므로 기본은 끔 |

## `ddoc doctor`

환경 점검. Python, 코어 라이브러리, 설치된 플러그인이 **실제로 불러지는지**, 선택 기능(PDF
보고서, `ddoc serve`, 임베딩 비교)을 쓸 수 있는지 봅니다. 플러그인이 설치되어 있어도 불러지지
않으면 그 플러그인이 맡는 데이터는 BLIND 가 되므로, 빠진 모듈을 함께 보여줍니다.
종료 코드: 코어가 동작하면 0, 코어에 필요한 것이 없으면 1. `./run.sh doctor` 는 여기에
venv 위치와 Docker·Chrome 유무를 더합니다.

| 옵션 | 설명 |
|---|---|
| `--json` | 점검 결과를 JSON 한 객체로 |
| `--lang` | `en` · `ko` |

## `ddoc probe <ref> [<cur>]`

`ddoc diff` 가 이 경로들을 어떻게 다룰지 판정 없이 미리 봅니다. 데이터 종류, 누가 판정할지(플러그인
또는 내장 CSV 비교기), BLIND 가 될지와 그 이유. CSV 두 개면 행 수, 비교할 열, 한쪽에만 있는 열,
날짜·시간 열도 보여줍니다. 종료 코드: 판정할 수 있으면 0, BLIND 가 되면 2.

```bash
ddoc probe jan.csv aug.csv
ddoc probe camA_ref/ camA_now/ --json
```

| 옵션 | 설명 |
|---|---|
| `--json` | 결과를 JSON 한 객체로 |
| `--lang` | `en` · `ko` |

## `ddoc scenarios`

상황별 명령 카탈로그(`ddoc/scenarios.conf`). 같은 목록을 `./run.sh menu`, `ddoc serve` 의 "상황별" 탭,
[치트시트](../cheatsheet.md)가 함께 씁니다.

```bash
ddoc scenarios                                   # 그룹별 목록
ddoc scenarios --md --lang ko                    # 치트시트 (Markdown)
ddoc scenarios run compare --set ref=jan.csv --set cur=aug.csv
ddoc scenarios run compare --set ref=a --set cur=b -n   # 실행하지 않고 명령만
ddoc scenarios menu                              # 번호로 고르고 값을 채워 실행
```

| 명령·옵션 | 설명 |
|---|---|
| `--md` / `--json` | 목록을 Markdown 치트시트 / JSON 으로 |
| `run <id>` `--set 이름=값` | 시나리오 실행. 값이 없는 필수 매개변수가 있으면 종료 코드 2 |
| `run <id>` `-n`, `--dry-run` | 실행하지 않고 명령만 출력 |
| `menu` | 번호로 고르기 → 값 입력 → 명령 확인 → 실행 |
| `--lang` | `en` · `ko` (실행하는 명령에도 같은 언어 적용) |

여러 줄짜리 시나리오는 차례로 실행하며, 마지막 줄을 뺀 단계가 실패하면 멈춥니다. 종료 코드는
마지막 명령의 것입니다(diff 면 0 OK · 1 DRIFT · 2 BLIND).

## `ddoc analyze drift`

플러그인 기반 저수준 drift 분석.

| 옵션 | 설명 |
|---|---|
| `--data-path-ref` / `--data-path-cur` | 기준 / 현재 데이터 경로 (둘 다 필수) |
| `--detector` | 검출 전략 |
| `--with-embeddings` | 임베딩 즉석 계산 (vision / text) |
| `--fusion` | `none` · `weighted` · `max` · `joint` |
| `--fusion-weights` | 예: `image=0.6,text=0.4` |
| `--json` / `--ndjson-progress` / `--quiet` | 출력 형식 제어 |

## `ddoc analyze eda`

`eda_run` 플러그인이 처리하지 않는 CSV 는 내장 프로파일러가 기술합니다 (규모, 기간,
열별 유형·분포, 품질 소견).

| 옵션 | 설명 |
|---|---|
| `--data-path` | 분석할 데이터 경로 (필수) |
| `--invalidate-cache` | 분석 캐시 무시 |
| `--sample-rows N` | 내장 CSV 프로필에 앞 N행 포함 (0–20, 기본 끔 — 원본 데이터) |
| `--max-rows N` | 내장 CSV 프로필: 이보다 크면 균등 무작위 표본 N행으로 (기본 200000, `0` = 전체) |
| `--report <path>` | 데이터 프로필 보고서도 함께 저장 (`.html` · `.md` · `.pdf`) |
| `--lang` | 보고서 언어 (`en` · `ko`) |
| `--json` / `--ndjson-progress` / `--quiet` | 출력 형식 제어 |

## `ddoc report render`

drift / EDA 결과를 리포트로 렌더. drift 입력은 `ddoc diff --report` 와 같은 진단서가
됩니다. `ddoc diff --json` 출력은 그대로 쓰고, `ddoc analyze drift --json` envelope 은
`ddoc diff` 와 같은 기준으로 판정을 먼저 계산합니다. Markdown 은 판정·판독·유형 카드·
계통별 검사표·점검 항목(체크박스)을 담아 PR 코멘트나 이슈에 붙일 수 있습니다. PDF 는
같은 HTML 을 인쇄한 것입니다. EDA 입력은 같은 형식의 데이터 프로필 보고서가 됩니다
(내장 CSV 프로필은 도면·품질 소견·열별 분포까지, 플러그인 EDA 는 요약을 같은 문서 안에).

```bash
ddoc diff ref.csv cur.csv --json > d.json
ddoc report render -i d.json -o d.md          # PR 코멘트용
ddoc report render -i d.json -o d.pdf --issuer "기관명"
```

| 옵션 | 설명 |
|---|---|
| `-i`, `--input` | `ddoc diff --json` 출력, drift envelope 또는 EDA envelope JSON 경로 (필수) |
| `-o`, `--out` | 출력 경로 (필수). 형식은 확장자로 추정 |
| `-f`, `--format` | `html` · `pdf` · `md` |
| `--title` | 리포트 제목 |
| `--lang` | 리포트 언어 (`en` · `ko`, 기본값 `$DDOC_LANG`, 없으면 `en`) |
| `--issuer` | 진단서 발행처 (기본값 `$DDOC_ISSUER`, 없으면 `ddoc`) |
| `--threshold` / `--attr-threshold` | drift envelope 을 판정할 기준 (기본 0.15 / 0.25, `ddoc diff` 와 같음) |
| `--json` | 결과 envelope 출력 |

## `ddoc export drift-report <input>`

drift 결과를 외부로 전송. 내장 adapter: `http`, `file`.

```bash
ddoc export drift-report drift.json --to file --config '{"out_dir":"/tmp/exports"}'
ddoc export drift-report drift.json --to http --config '{"base_url":"http://my-app.local:8000"}'
```

| 옵션 | 설명 |
|---|---|
| `-t`, `--to` | 대상 adapter (필수) |
| `--config` / `--config-file` | 대상별 설정 (JSON 문자열 / YAML·JSON 파일) |
| `--json` | 결과 envelope 출력 |

## `ddoc fetch <source_uri>`

외부 데이터를 로컬 디렉토리로 가져오기. 내장: `file://` 와 일반 경로.
`s3://`, `gs://`, `http(s)://` 등은 플러그인이 제공합니다.

| 옵션 | 설명 |
|---|---|
| `-d`, `--dest` | 저장할 디렉토리 (필수) |
| `--symlink` | file 소스를 복사 대신 symlink |
| `--config` | adapter 설정 JSON |
| `--json` | 결과 envelope 출력 |

## `ddoc recipe`

| 하위 명령 | 설명 |
|---|---|
| `run <file>` | 레시피의 모든 단계 실행 (`--dry-run`, `--json`) |
| `validate <file>` | 실행 없이 파싱·검증 |
| `kinds` | 레시피에서 쓸 수 있는 단계 종류 목록 |

## `ddoc serve`

모든 명령을 HTTP 로 노출하는 FastAPI 서버 + 브라우저 GUI.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--host` | 127.0.0.1 | 0.0.0.0 은 `--api-key` 와 TLS 뒤에서만 |
| `--port` | 8765 | |
| `--api-key` | | `X-API-Key` 헤더로 요구할 키 (`DDOC_API_KEY` 환경변수보다 우선) |
| `--log-level` | info | |
| `--reload` | | 개발용 자동 재시작 |

브라우저 화면(진단 화면)의 "서버에서 찾기"는 `DDOC_DATA_ROOT`(기본: 서버를 띄운 폴더와 홈 폴더, 여러 개는 `:` 로
구분) 아래만 보여주고, 올린 파일은 `DDOC_WORKSPACE`(기본: 임시 폴더)에 하루 동안 둡니다. 파일 하나의 크기는
`DDOC_UPLOAD_MAX_MB`(기본 1024)까지입니다. 자세한 내용은 [REST / GUI](../tutorial/serve.md#browser-gui).

키가 없으면 이 컴퓨터만 상대합니다. `Host` 가 localhost · 127.0.0.1 · ::1 · 바인딩 주소 · `DDOC_ALLOWED_HOSTS`
(쉼표로 구분) 가 아닌 요청과, 다른 사이트에서 온(`Origin` 이 다른) 변경 요청은 403 입니다. 키를 설정하면
키가 이 역할을 맡습니다.

## `ddoc plugin`

| 하위 명령 | 설명 |
|---|---|
| `list` | 설치된 플러그인 목록 |
| `info` | 플러그인 상세 정보 |
| `install` | pip 로 플러그인 설치 |
| `detectors` | 플러그인별 지원 검출기 |

## `ddoc examples`

| 하위 명령 | 설명 |
|---|---|
| `list` | 지원하는 (modality, scenario) 조합 |
| `generate <modality> -o <dir> [-s shifted\|identical]` | `ref/`, `cur/` toy 데이터 생성 |

## `ddoc transform`

플러그인의 `transform_apply` hook 실행.

| 옵션 | 설명 |
|---|---|
| `--input-path` / `--output-path` | 입력 / 출력 경로 (필수) |
| `--transform` | 플러그인이 제공하는 transform 이름 (필수) |
| `--args-json` | transform 인자 JSON |
| `--json` | 결과 envelope 출력 |
