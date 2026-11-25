# 치트시트 — 상황별 명령

`ddoc/scenarios.conf` 에서 만든 문서입니다. 고치려면 그 파일을 고친 뒤 `./run.sh docs build`.
같은 목록을 `ddoc scenarios`(터미널), `./run.sh menu`(번호로 고르기), `ddoc serve` 의 "상황별" 탭이
함께 씁니다.

<!-- generated:cheatsheet — scripts/docs.py build -->
상황마다 그대로 칠 명령입니다. `ddoc scenarios run <id> --set 이름=값` 으로 바로 실행하고, `-n` 을 붙이면 실행하지 않고 명령만 보여줍니다. `./run.sh menu` 는 번호로 골라 실행합니다.

### 처음 해보기

- **환경 점검** (`check-env`) — 처음 시작할 때, 또는 예상과 달리 BLIND 가 나올 때  
  `ddoc doctor`  
  - ✕ 표시 플러그인은 설치는 되어 있지만 불러지지 않습니다 — 그 데이터는 BLIND 가 됩니다
- **투어로 한 번에 보기** (`tour`) — 실제 데이터로 파이프라인 전체를 한 단계씩: 왜 하는지, 명령, 그 출력  
  `ddoc examples tour weather`  
  - 파일은 ./ddoc-tour-weather 에 남습니다. --pause 는 단계마다 Enter 를 기다립니다
  - 투어는 레시피 한 장입니다: --save my.yaml 로 꺼내 ddoc recipe run my.yaml --narrate 로 내 데이터에
- **예제 데이터로 해보기** (`try-demo`) — 처음 한 번: 드리프트가 있는 예제 쌍을 만들어 판정하고 진단서를 엽니다  
  `ddoc examples generate categorical --out ddoc-try --scenario shifted`  
  `ddoc diff ddoc-try/ref ddoc-try/cur --report ddoc-try/note.html`  
  - 브라우저로 ddoc-try/note.html 을 여세요. 종료 코드 1 이 DRIFT 입니다
  - categorical 플러그인이 필요합니다 (./run.sh setup 이 설치)

### 두 데이터 비교

- **판정 전에 미리 보기** (`look-first`) — diff 가 두 경로를 어떻게 다룰지 확인: 데이터 종류, 판정 주체, BLIND 여부  
  `ddoc probe <ref> <cur>`  
  - CSV 두 개면 비교할 열과 한쪽에만 있는 열도 보여줍니다
- **판정하고 진단서 만들기** (`compare`) — 가장 흔한 경우: 판정과 그 근거를 담은 진단서  
  `ddoc diff <ref> <cur> --report note.html`  
  - 종료 코드: 0 OK · 1 DRIFT · 2 BLIND (판정 불가)
  - 한국어 진단서는 DDOC_LANG=ko (또는 --lang ko)
- **OK 인데 의심스러울 때** (`suspect-ok`) — 판정은 OK 인데 변화가 있을 것 같을 때 — 예: 값 종류가 적은 열  
  `ddoc diff <ref> <cur> --attr-threshold 0.15 --report note.html`  
  - 유형 판독의 "잠복 변화형" 과 "가까움" 목록을 보세요
  - 기준을 낮추면 더 많이 잡힙니다. 측정하기 전에는 CI 에서는 기본값을 쓰세요
- **이미지·텍스트를 임베딩 지도와 함께** (`compare-images`) — 이미지·텍스트 폴더에서 내용 자체가 바뀌었을 수 있을 때  
  `ddoc diff <ref> <cur> --with-embeddings --report note.html`  
  - torch·CLIP 과 vision 또는 text 플러그인이 필요합니다 (`ddoc doctor` 로 확인)
  - 지도의 이웃 섞임이 0 에 가까우면 두 데이터가 갈라진 것입니다 — 각 중심 근처 항목을 보세요
- **앞 몇 행을 넣은 진단서** (`with-samples`) — 읽는 사람이 실제 행 모양을 봐야 할 때 (공유 범위 주의)  
  `ddoc diff <ref> <cur> --sample-rows 3 --report note.html`  
  - 들어가는 행은 원본 데이터입니다 — 보낼 대상을 확인하세요

### 데이터 하나

- **데이터 하나 살펴보기** (`profile`) — 비교 전에 CSV 하나의 규모·기간·열·품질 소견 보기  
  `ddoc analyze eda --data-path <data_path> --report profile.html --quiet`  
  - 식별자형 열과 날짜 열은 드리프트 비교에서 빼는 편이 좋습니다

### 공유와 자동화

- **PR·이슈에 붙일 Markdown** (`pr-comment`) — 판정·판독·점검 체크리스트를 PR·이슈·채팅에 붙일 때  
  `ddoc diff <ref> <cur> --report drift.md`  
  - 먼저 할 일이 체크박스(- [ ])로 들어갑니다
- **CI 데이터 게이트** (`ci-gate`) — 들어온 데이터가 드리프트면 파이프라인을 멈출 때  
  - 종료 코드가 곧 판정입니다: 0 OK, 1 DRIFT, 2 BLIND — `ddoc diff ref cur` 하나가 게이트
  - GitHub Actions: uses: keti-datadrift/datadrift_base_framework@main (examples/ci-gate 참고)
- **저장한 결과로 보고서 다시 만들기** (`render-again`) — JSON 결과(diff --json, analyze --json)를 HTML·PDF·Markdown 으로 다시 만들 때  
  `ddoc report render -i <input> -o report.html`  
  - drift envelope 은 diff 와 같은 기준으로 판정됩니다 (--threshold, --attr-threshold)
- **브라우저 GUI 와 REST API** (`serve`) — 브라우저에서 진단하거나 다른 서비스에서 ddoc 을 부를 때  
  `ddoc serve --port 8765`  
  - http://127.0.0.1:8765 를 여세요 — 진단 탭에서 진단서를 바로 봅니다
  - localhost 밖으로 열 때는 DDOC_API_KEY 를 먼저 설정하세요
<!-- /generated:cheatsheet -->
