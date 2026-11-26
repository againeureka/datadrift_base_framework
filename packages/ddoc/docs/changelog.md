# Changelog - ddoc

> **참고**: 각 버전의 상세 릴리스 노트는 [releases/](releases/) 디렉토리에서 확인할 수 있습니다.

## [3.0.0] - Unreleased

### 🩺 판정 도구로 집중 — 하위 호환이 깨지는 릴리스

**주요 변경사항:**
- ⚖️ **판정 점수를 바꿈** — 속성마다 JS 거리를 내고, 같은 구간의 G-검정에서 표본 잡음으로 설명되면 0. 잡음을 넘은
  유형 판독은 혼자서도 DRIFT(결측 증가·소수의 먼 극단값처럼 분포 점수에 안 드러나는 변화). CSV 열, 플러그인이
  잰 값(vision·text·audio·timeseries), categorical 개수, 임베딩 지도(`embedding_separation`)를 모두 같은 0–1
  척도로 매김. 드릴·이야기 21/21(이전 11/21), 같은 분포 30행 표의 오경보 97% → 1.5%. 점수 숫자가 커지고
  (발산 0.06 → 거리 0.25) 잠복 변화형의 뜻이 "조건은 채웠지만 잡음 범위 안"으로 바뀜 ([판정 민감도](sensitivity.md))
- ⚖️ **시간 순서 기록은 하루 단위로 잡음 검정** — 시간 열(timeseries 플러그인은 `timestamp_column`)이 있으면 행을
  날짜로 묶어 두 기간의 차이가 날마다의 차이를 넘는지 봄(2차 Rao–Scott 보정). 같은 날씨의 두 기간을 행 단위로는
  71–95% DRIFT 로 판정했고, 하루 단위로는 0–14%. 실측 기상 기록의 연간 초기화·나흘 대 나흘은 OK, 장마·계절은 DRIFT
- ✨ **0점의 이유를 보여 줌** — 진단서·진단 화면이 0점마다 "잡음 범위 안 (거리 0.312, p 0.25, 하루 단위로 셈)"인지
  "분포 같음"인지, 결측만 달라졌는지 적고, 잡음 범위 안의 열은 받았을 거리를 점선으로 그림. OK 판정의 근거에
  "N개 속성이 달라 보이지만 표본 잡음 범위 안"이 더해지고, 콘솔은 잡음을 넘었다면 기준에 닿았을 열을
  "잡음 범위 안" 줄로 보여 줌. Markdown 진단서에 판정 근거 절. 결과 JSON 의 `diagnosis.noise` 와 요약의 `distance`
- ⚖️ **evidently 도 같은 척도** — `ddoc diff --detector evidently:*` 의 판정을 같은 CSV 의 내장 비교기 점수로 내리고
  Evidently 의 열별 검정은 `details`·`plugin_scores` 에 남김. 거리 검정(Wasserstein·Jensen–Shannon)의 값을 p값으로
  읽어 크게 바뀐 열이 0점이 되던 플러그인 버그 수정
- ✨ `ddoc diff` 한 줄 판정 (OK / DRIFT / BLIND), `--report` HTML 진단서, Drift Zoo, GitHub Action
- ⚡ 큰 CSV 는 쪽마다 균등 무작위 표본(기본 20만 행, `--max-rows`)으로 비교·프로필, 진단서에 표시
- ✨ 임베딩 지도에 열어 볼 항목(기준·현재의 대표 항목, 기준과 가장 다른 현재 항목)을 파일 이름으로
- ✨ 날짜·시간 열을 이름뿐 아니라 값의 모양(연도 네 자리 날짜·ISO 시각)으로도 찾음
- 📦 vision·text 플러그인이 CLIP 을 패키지에 담음 (GitHub 의존성 제거, 어휘 파일을 wheel 에 포함)
- 📦 text 플러그인이 nltk 없이 동작 — 내려받을 말뭉치가 없어 오프라인·프록시 뒤에서도 실행 (단어는 정규식,
  영어 불용어는 scikit-learn 목록). 최근 nltk 에서 `punkt_tab` 이 없어 실패하던 문제도 함께 해결
- 📝 [지금 상태](status.md) (확인한 방법과 알려진 한계) · [용어](glossary.md)
- 📝 [사용 예제](guides/usage.md) 한국어·[영어](guides/usage.en.md) — 상황별 예제와 실제 출력, 저장소 README 에서 연결.
  예제의 명령·옵션이 실제 CLI 에 있는지, 두 언어판이 같은 명령을 보여주는지, 링크가 살아 있는지 테스트가 확인
- ✨ 판독 드릴 — 유형마다 정답이 있는 케이스, `./run.sh zoo` 와 테스트로 확인, 결과표는 zoo README 에 생성. vision 예제를 도형·색이 다양한 40장으로 교체
- ✨ 상황별 시나리오 카탈로그 — `ddoc scenarios` (목록·치트시트·실행·메뉴), `./run.sh menu`, GUI "상황별" 탭, 생성되는 치트시트. `diff --report` 의 `.md`·`.pdf`, `analyze eda --report`
- ✨ `ddoc doctor` (환경·플러그인 로드·선택 기능 점검), `ddoc probe` (diff 전에 데이터 종류·판정 주체·BLIND 여부 확인), `./run.sh doctor`
- ✨ 내장 CSV 프로파일러(`analyze eda`)와 데이터 프로필 보고서 (`--sample-rows`)
- ✨ `report render` 가 drift 입력을 진단서(HTML·PDF·Markdown)로 렌더, `ddoc serve` 의 `POST /diff` 와 GUI 진단 탭
- ✨ `ddoc serve` 의 첫 화면을 진단 화면으로 — 파일·폴더 끌어다 놓기와 서버 폴더 탐색으로 데이터를 넣고, 넣는 즉시
  내용(행·열·기간·열 이름)과 판정 전 확인을 보여주며, 결과는 판정·유형 카드·열별 분포 그림·진단서(HTML·PDF·Markdown
  저장)로. 정답이 있는 예제 사례 4개, 주소로 바로 여는 링크(`/?case=…`, `/?ref=…&cur=…`), 주간·야간. 기존 폼은 "고급 도구"로
- ✨ 누적·순번 열(일·월·연 누적 강우, 누적 횟수, 순번)은 날짜 열처럼 비교에서 빼고 이름을 적음 — 시간 열이 있을 때
- ✨ 실측 예제 데이터: 가정용 기상관측소 기록. Drift Zoo [weather_station](../zoo/weather_station/), 진단 화면 예제 사례 2개,
  `examples generate timeseries`
- ✨ [투어](tutorial/tours.md) — `ddoc examples tour weather`: 실제 데이터로 받기 → 미리 보기 → 살펴보기 → 판정·진단서 →
  DRIFT 일 때만 공유까지, 단계마다 이유·명령·그 명령의 출력. 레시피 한 장이라 `--save` 로 내 데이터의 파이프라인이 됨.
  `./run.sh demo` 가 이 투어로
- ✨ 실측 이미지 예제: 블랙박스 주행 12분의 프레임(전방·후방 카메라, 축소·대시보드 아래 잘라냄). Drift Zoo
  [dashcam](../zoo/dashcam/), 투어 `ddoc examples tour dashcam`, 진단 화면 예제 사례 3개, `examples generate dashcam`
- 🐛 vision 플러그인이 있으면 `analyze eda` 가 CSV 대신 빈 이미지 분석 결과를 내던 문제, 레시피의 플러그인 설치
  안내가 PyPI 이름이던 문제
- 🐛 플러그인으로 비교하면 실행한 폴더에 `.ddoc/`·`analysis/`, 이미지 폴더 안에 `cache/` 가 남던 문제 — 캐시는
  실제로 저장할 때만 만들고, 데이터 해시가 없는 경로 비교에서는 저장하지 않음
- ✨ 생성 텍스트·오디오 예제: 메모 앱 고객 문의([support_tickets](../zoo/support_tickets/): 앱 업데이트 · 채팅 말투 ·
  평범한 두 주), 모터 소리([machine_sound](../zoo/machine_sound/): 베어링 마모 · 마이크 이득 · 두 교대). 투어
  `ddoc examples tour support|machine`, 진단 화면 예제 사례 3개, `examples generate support|machine`
- ✨ 이미지·텍스트·오디오도 속성 판독과 분포 그래프 — 플러그인이 항목마다 잰 값을 넘기면 CSV 의 수치 열과 같은
  방식으로 읽음 (점수는 플러그인 것 그대로)
- ✨ 레시피에 `diff`·`probe` 단계(판정은 실패가 아님), `${steps.<id>.verdict}`·`returncode`, 단계의 `title`·`note`,
  `recipe run --narrate`·`--var`. `examples generate weather` (실측 기상 기록, 이야기 이름으로)
- 🎨 `analyze eda` 의 화면 출력이 결과 JSON 전체 대신 몇 줄 요약 (크기·열 종류·기간·품질 소견). 전체는 `--json`·`--report`
- ✨ 이동·수축형 판독 — 평균이 옮겨가면서 폭이 좁아진 열 (한계값 쪽으로 몰림, 좁은 집단만 남음). 장마의 이슬점처럼
  가장 크게 변한 열이 어느 유형에도 들지 않던 빈칸을 메움
- 🐛 기준을 넘은 속성이 8개를 넘으면 진단서가 만들어지지 않던 문제, 진단서 실패를 알리지 않던 문제, timeseries
  플러그인의 NaN 점수
- 🎨 진단서·프로필 보고서·진단 화면의 강조 표시를 왼쪽 굵은 띠 대신 판독 낫표 「 」(유형 카드·판정)와 여백 ※(주의)로.
  표지마다 뜻이 하나 ([개념](tutorial/concepts.md#5-진단서와-진단))
- ✨ `ddoc serve` (REST + GUI), `recipe`, `report`, `export`, `fetch`, `examples` 명령
- ✨ 한글 리포트 (`--lang ko` / `DDOC_LANG=ko`)
- ✨ 검사 보고서 형식의 진단서: 발행처·문서 번호, 요약 대시보드, 속성별 그래프, 데이터 특징, 패턴별 점검 항목 (`--json` 의 `diagnosis`), 화면 테마(자동·주간·야간)와 인쇄 버튼, 검사 대상 섹션, 유형 판독(18가지 드리프트 유형 카드), 임베딩 지도(`--with-embeddings`), 데이터 도면, 계통별 검사표, `--sample-rows`
- 🗑️ `init` / `add` / `snapshot`, 스냅샷 모드, `exp` / `train`, `ingest`, `vis`, v1 legacy 명령 제거
- 🗑️ yolo / vis / nlp / reference-engine / keti-temporal 플러그인, 관련 hookspec 과 extras 제거 (hookspec 2.0.0)
- 🔒 `ddoc serve` 를 키 없이 띄우면 이 컴퓨터만 상대 (낯선 `Host` 와 다른 사이트의 변경 요청은 403,
  `DDOC_ALLOWED_HOSTS`), 키 비교를 상수 시간으로, `-` 로 시작해 옵션으로 읽힐 경로 값은 거절
- 📝 [판정 민감도](sensitivity.md) — 점수 방식 후보별 판정·오경보·검출 한계 비교 (기본값은 그대로)
- 🐛 CSV 의 날짜·시간 텍스트 열을 비교하던 문제 — 이제 기간 표시에만 사용
- 🐛 `ddoc serve` 가 JSON 이 아닌 요청 본문에 422 대신 500 을 내던 문제
- 🐛 플러그인 설치 안내가 PyPI 이름(`pip install ddoc-plugin-vision`, `ddoc-full`)이라 실패하던 문제 — 아직 PyPI 에
  없으므로 저장소 폴더(`pip install -e …/ddoc-plugin-vision`)나 git 주소로 안내
- 🐛 `ddoc diff` 가 실행한 디렉터리에 `analysis/diff/metrics.json` 을 남기던 문제
- 🐛 PDF 진단서·보고서를 쓸 때 PDF 엔진 로그 수십~수천 줄이 stdout 에 섞여 `--json` 출력이 깨지던 문제 (로그는 stderr 로)
- 🐛 플러그인 하나의 실패가 다른 플러그인 결과를 지우던 문제, 일반 설치에서 `examples generate` 가 실패하던 문제 수정
- 🔄 코어와 플러그인을 `packages/ddoc/`, `packages/plugins/` 로 이동

> 📖 **상세 내역**: [v3.0.0 릴리스 노트](releases/v3.0.0.md)

---

## [2.1.0] - 2025-12-30

### 🎯 Multi-Modal Analysis - 멀티모달 분석 기능 추가

**주요 변경사항:**
- ✨ 이미지/텍스트/시계열/오디오 멀티모달 분석 지원
- ✨ 이미지 속성 지표 9종 구현 및 속성 드리프트 개선
- ✨ 텍스트 분석 플러그인 신규 추가 (CSV + CLIP 텍스트 임베딩)
- ✨ 시계열 분석 플러그인 신규 추가 (숫자형/범주형 지표)
- ✨ 오디오 분석 플러그인 신규 추가 (물리 기반 특징)
- ✨ 캐시 네임스페이싱으로 모달리티별 독립 캐시 관리
- ✨ ddoc-full 메타패키지로 모든 플러그인 통합 설치
- 🗑️ Builtin 텍스트 단일파일 fallback 훅 제거

> 📖 **상세 내역**: [v2.1.0 릴리스 노트](releases/v2.1.0.md)

---

## [2.0.4] - 2025-12-23

### 🧩 Snapshot 안정화 및 CLI 개편

**주요 변경사항:**
- 🐛 초기 워크스페이스에서 스냅샷 생성 실패(`data.dvc not found`) 문제 해결
- 🐛 스냅샷 체크아웃 시 `.ddoc` 변경으로 인해 Git checkout이 실패하는 문제 개선
- 🧹 체크아웃 후 `data/` 하위 빈 디렉터리 및 `.DS_Store` 잔재 자동 정리
- 🔄 `snapshot` 명령을 하이브리드 구조로 개편 (핵심 서브커맨드 + 부가 옵션)

> 📖 **상세 내역**: [v2.0.4 릴리스 노트](releases/v2.0.4.md)

---

## [2.0.3] - 2024-12-04

### 🎯 Experiment 명령 개편 - Trainer 기반 실험 시스템

기존 YOLO 전용 `exp run` 명령을 범용 Trainer 기반 시스템으로 완전히 개편했습니다.

**주요 변경사항:**
- ✨ Trainer 기반 실험 시스템 (`ddoc exp train`, `ddoc exp eval`, `ddoc exp best`)
- ✨ Trainer 인터페이스 표준화 (`code/trainers/{name}/train.py`, `eval.py`)
- 🔄 `ddoc exp run` → `ddoc exp train`으로 대체 (deprecated)
- 📦 `ddoc/core/trainer_service.py` 신규 모듈 추가

> 📖 **상세 내역**: [v2.0.3 릴리스 노트](releases/v2.0.3.md)

---

## [2.0.2] - 2025-11-25

### 🎯 워크스페이스 캐시 동기화 및 증분 분석 안정화

**주요 변경사항:**
- ✨ Workspace Cache Synchronization (자동 캐시 동기화)
- 🐛 증분 분석 빈 캐시 문제 해결 (Critical)
- 🐛 Drift 분석 캐시 타입 오류 수정
- 📦 하이브리드 캐시 구조 (파일 시스템 + SQLite)

> 📖 **상세 내역**: [v2.0.2 릴리스 노트](releases/v2.0.2.md)

---

## [2.0.1] - 2025-11-17

### 🚀 성능 및 개발자 경험 개선

**주요 변경사항:**
- ⚡ CLI 시작 속도 21배 향상
- ⚡ 플러그인 목록 조회 21.8배 향상
- 📁 CLI 모듈화 (2,767줄 → 평균 120줄)
- 🛠️ 압축 파일 해제 수정

> 📖 **상세 내역**: [v2.0.1 릴리스 노트](releases/v2.0.1.md)

---

## [2.0.0] - 2025-11-14

### 🎉 Major Release - Complete System Overhaul

ddoc v2.0.0은 완전히 새로운 아키텍처로 재설계되었습니다.

**주요 변경사항:**
- ✨ Git-like 워크플로우 (commit, checkout, log, diff)
- ✨ Snapshot-based 버전 관리
- ✨ 자동 프로젝트 스캐폴딩
- ⚠️ Breaking Changes: 명령어 구조 전면 변경

> 📖 **상세 내역**: [v2.0.0 릴리스 노트](releases/v2.0.0.md)

---

**전체 릴리스 노트**: [releases/](releases/) 디렉토리 참조

