# 용어

진단서·명령·JSON 에 나오는 말입니다. 괄호 안은 영어 화면과 JSON 에서 쓰는 이름입니다.

## 판정

| 용어 | 뜻 |
|---|---|
| 기준 데이터 (baseline, `ref`) | 정상으로 보는 쪽. 비교의 기준 |
| 현재 데이터 (current, `cur`) | 기준과 얼마나 달라졌는지 볼 쪽 |
| 판정 (verdict) | `OK` · `DRIFT` · `BLIND` 셋 중 하나. 종료 코드가 0 · 1 · 2 |
| BLIND | 판정할 수 없음. 데이터가 없거나, 종류가 다르거나, 맡을 플러그인이 없을 때. 조용한 OK 대신 내는 정직한 답 |
| 심각도 (severity) | DRIFT 의 정도 (`medium` · `high`), OK 의 정도 (`none` · `low`) |
| 속성 (attribute) | 비교 단위. CSV 의 열, 이미지의 밝기·대비 같은 측정값, 범주 분포 하나 |
| 점수 (score) | 속성마다 기준과 현재 분포의 JS 거리 (0 같음, 1 겹침 없음). 표본 잡음으로 설명되는 차이는 0. 모든 데이터 종류에 같은 척도 |
| 잡음 검정 (noise test) | 같은 구간의 G-검정. p ≥ 0.001 이면 그 차이는 표본 우연으로도 생길 수 있다고 보고 점수를 0 으로, 유형 판독은 "가까움"으로 둠 |
| 전체 점수 (overall score) | 속성 점수의 평균 |
| 기준 · 단일 속성 기준 (`--threshold`, `--attr-threshold`) | 전체 점수가 기준(0.15) 이상이거나 한 속성이 단일 속성 기준(0.25) 이상이면 DRIFT. 잡음을 넘은 유형 판독도 DRIFT |

## 진단서

| 용어 | 뜻 |
|---|---|
| 진단서 (diagnostic report) | `ddoc diff --report` 가 쓰는 보고서. HTML · Markdown · PDF |
| 유형 판독 (pattern reading) | 측정값을 미리 정리해 둔 드리프트 유형의 조건에 대어 본 결과 |
| 해당 · 가까움 (match · near) | 조건을 모두 채움 · 하나가 근소하게 모자람 (조기 신호) |
| 판단 기준 (read when) | 유형마다 "언제 그렇게 읽는가"를 적은 문장. 코드의 기준값에서 만들어짐 |
| 계통별 검사표 (inspection by system) | 계통(구성·표본·분포·완전성·수치·범주)별 항목의 측정값, 기준, 판정 |
| 적합 · 주의 · 부적합 (pass · caution · fail) | 검사표 판정. 부적합은 판정을 만든 점수, 주의는 기준을 넘은 변화 패턴 |
| 데이터 도면 (data diagram) | 열 하나를 부품 하나로 그린 그림. 살펴볼 부품에 번호 |
| 임베딩 지도 (embedding map) | 이미지·텍스트 임베딩을 2차원에 찍은 산점도 |
| 중심 이동 (centre shift) | 현재 중심이 기준 퍼짐의 몇 배만큼 움직였는지 |
| 이웃 섞임 (neighbour mixing) | 현재 항목 주변에 기준 항목이 섞여 있는 정도. 1 = 고르게 섞임, 0 = 완전히 분리 |
| 발행처 · 문서 번호 (issuer · document no.) | 진단서 머리글. `--issuer`, `DD-날짜-해시` |

## 데이터 하나

| 용어 | 뜻 |
|---|---|
| 데이터 프로필 보고서 (data profile report) | `ddoc analyze eda --report` 가 쓰는, 데이터 하나를 기술하는 보고서 |
| 품질 소견 (quality findings) | 결측, 상수 열, 식별자형 열, 극단값, 표기 변형 등 데이터 하나의 문제 |
| 식별자형 열 (identifier-like) | 거의 모든 값이 다른 열 (ID, 키). 분포 비교에 쓰기 어려움 |
| 표기 변형 (spelling variant) | 대소문자·공백만 다른 값 (`Store` ≈ `store`) |

## 구조

| 용어 | 뜻 |
|---|---|
| 플러그인 (plugin) | 데이터 종류별 분석을 맡는 별도 패키지 (`ddoc-plugin-vision` 등) |
| 검출기 (detector) | 플러그인 안의 비교 방법 (`--detector`) |
| 데이터 종류 (modality) | csv · categorical · vision · text · audio … `ddoc probe` 가 알려줌 |
| envelope | 명령이 `--json` 으로 내는 결과 객체. 진단은 `diagnosis`, 원 결과는 `raw` |
| Drift Zoo | 유명한 드리프트 이야기를 재현하는 케이스 모음 ([zoo](../zoo/)) |
| 판독 드릴 (drill) | 유형마다 변화 하나만 심고 정답을 정해 둔 케이스 |
| 시나리오 (scenario) | 상황별 명령 카탈로그의 항목 ([치트시트](cheatsheet.md)) |

## 기호

| 기호 | 뜻 |
|---|---|
| σ | 기준 데이터의 표준편차. "평균 0.5σ 이동" = 기준 표준편차의 절반만큼 이동 |
| %p | 퍼센트포인트. 20% → 30% 는 10%p |
| IQR | 사분위 범위 (상위 25% 와 하위 25% 사이의 폭) |
