# Toy data — 5분 안에 첫 drift 결과 보기

ddoc 를 처음 만지는 사용자가 본인 데이터 없이도 **drift 가 어떻게
계산되고 어떤 envelope 으로 떨어지는지** 5 분 안에 직접 확인할 수
있도록 5 modality (timeseries / audio / text / vision / categorical) 의
예제 데이터셋을 generator 로 제공합니다. timeseries 는 실제 기상관측소 기록이고, 나머지는 합성
데이터입니다.

같은 generator 를 pytest 회귀 (`tests/test_plugin_drift_e2e.py`) 도
사용 — 즉 사용자가 `ddoc examples generate` 로 만드는 데이터는 ddoc
자체 테스트가 검증한 것과 byte-for-byte 동일합니다.

## 사용 가능한 modality 와 scenario

```bash
ddoc examples list
```

| modality | data shape | shifted scenario |
|---|---|---|
| `timeseries` | CSV `time` + 기상 측정값 11열 — 실제 가정용 기상관측소의 10분 기록 2주 (2,016행) | 장마 전 2주 → 장마가 시작된 2주 (이슬점·습도 상승) |
| `audio` | mono WAV, 8 kHz, 1 초 | 440 Hz → 880 Hz (octave) |
| `text` | CSV `id,text` (20 rows) | 짧은 단어 → 긴 문장 |
| `vision` | 40 × 96×96 PNG (밝은 배경에 색·크기가 제각각인 도형) | 어둡게, 푸른 톤 (카메라·조명이 바뀐 경우) |
| `categorical` | `distributions.json` — color/type/hourly counts + confidence stats | 차량 통계 형태의 분포, 4 분포 모두 shift |

각 modality 는 두 scenario 지원:
- `shifted` (default): ref 와 cur 분포가 다름 → drift score > 0
- `identical`: ref 와 cur 가 완전 동일 → drift score ≈ 0 (sanity check)

## 5분 quick start

```bash
# 1. timeseries 예제 데이터 생성 (drift 시나리오)
ddoc examples generate timeseries --out /tmp/ex --scenario shifted

# 2. drift 측정
ddoc analyze drift \
    --data-path-ref /tmp/ex/ref \
    --data-path-cur /tmp/ex/cur \
    --json --quiet
```

출력 (envelope):
```json
{"modality": "timeseries", "timestamp": "20261125_225058", "overall_score": 9403007.749453789}
```

`identical` scenario 로 다시 돌리면 score 는 0 입니다. 점수가 이렇게 큰 것은 timeseries 플러그인이 열의 원래
단위로 평균·분산·왜도·첨도의 차이를 더하기 때문입니다. 일사량(lux)처럼 값이 큰 열의 분산 차이가 점수를
좌우하므로, 이 숫자는 0–1 점수가 아니고 다른 플러그인과 비교할 수도 없습니다([지금 상태](../status.md)의
알려진 한계).

같은 CSV 를 `ddoc diff` 로 비교하면 0–1 점수, 판정, 가장 변한 열과 판독을 볼 수 있습니다:

```bash
ddoc diff /tmp/ex/ref/weather/data.csv /tmp/ex/cur/weather/data.csv --lang ko
# ⚠ DRIFT — 심각도: 높음 · 점수 0.1782 (기준 0.15)
#    가장 변한 속성: dew_point_c 0.909 · humidity 0.276 · pressure_inhg 0.237
```

데이터와 다른 기간의 이야기(연간 누적값 초기화, 계절 등)는 Drift Zoo 의
[weather_station](../../zoo/weather_station/) 에 있습니다.

## 다른 modality

플러그인은 아직 PyPI 에 올라가 있지 않아 `pip install ddoc-plugin-<이름>` 으로는 설치되지 않습니다.
저장소 루트에서 폴더 경로로 설치하거나(`pip install -e packages/plugins/ddoc-plugin-<이름>`) `./run.sh setup <이름>`
으로 `./.venv` 에 더합니다. 저장소를 받지 않았다면 git 주소로 설치합니다:
`pip install "git+https://github.com/keti-datadrift/datadrift_base_framework#subdirectory=packages/plugins/ddoc-plugin-<이름>"`.

audio 는 plugin (`ddoc-plugin-audio`) 과 librosa 가 필요합니다:

```bash
pip install -e packages/plugins/ddoc-plugin-audio
ddoc examples generate audio --out /tmp/au --scenario shifted
ddoc analyze drift --data-path-ref /tmp/au/ref --data-path-cur /tmp/au/cur \
    --json --quiet
```

text 는 `ddoc-plugin-text` (따로 내려받을 데이터 없음, `--with-embeddings` 를 처음 쓸 때 CLIP 가중치만 받음):

```bash
pip install -e packages/plugins/ddoc-plugin-text
ddoc examples generate text --out /tmp/tx --scenario shifted
ddoc analyze drift --data-path-ref /tmp/tx/ref --data-path-cur /tmp/tx/cur \
    --json --quiet
```

vision 은 `ddoc-plugin-vision` (torch·torchvision·pillow 가 함께 설치되고, CLIP 은 플러그인에 들어 있음):

```bash
pip install -e packages/plugins/ddoc-plugin-vision
ddoc examples generate vision --out /tmp/vi --scenario shifted
ddoc analyze drift --data-path-ref /tmp/vi/ref --data-path-cur /tmp/vi/cur \
    --json --quiet
```

categorical 은 *별도 모델 의존 없음* — 분포 dict 만 비교 (jensen-
shannon / overlap). vehicle / IoT / 카메라 분류 데이터에 맞는 modality:

```bash
pip install -e packages/plugins/ddoc-plugin-categorical    # ./run.sh setup 이 기본으로 설치
ddoc examples generate categorical --out /tmp/cat --scenario shifted
ddoc analyze drift --data-path-ref /tmp/cat/ref --data-path-cur /tmp/cat/cur \
    --json --quiet
```

생성되는 `distributions.json` 은 실제 카메라 통계 윈도우 출력과
동일한 shape — 즉 이 demo 는 categorical drift 파이프라인의 회귀
테스트로도 작용합니다.

## Tabular EDA report 렌더 (별도 path)

`ddoc-plugin-tabular` 는 *drift 검출* 이 아니라 *EDA
envelope 의 PDF/HTML 렌더* 를 담당합니다. 이미 만들어진 envelope
JSON (`{name, rows, cols, missing, summary}` 형태의 EDA dict) 을 받아
컬럼별 summary / missing-rate 표를 포함한 보고서로 출력합니다.

```bash
# 1. 샘플 envelope 작성 (실제로는 backend 가 emit; 여기선 손으로)
cat > /tmp/eda.json <<'EOF'
{
  "modality": "tabular",
  "name": "iris.csv",
  "rows": 150,
  "cols": 5,
  "missing": {"sepal_len": 0.0, "petal_len": 0.0},
  "summary": {
    "sepal_len": {"count": 150, "mean": 5.84, "std": 0.83,
                   "min": 4.3, "50%": 5.8, "max": 7.9},
    "petal_len": {"count": 150, "mean": 3.76, "std": 1.76,
                   "min": 1.0, "50%": 4.35, "max": 6.9}
  }
}
EOF

# 2. 렌더 (HTML)
ddoc report render -i /tmp/eda.json -o /tmp/iris.html --format html
# 또는 PDF (weasyprint 필요)
ddoc report render -i /tmp/eda.json -o /tmp/iris.pdf --format pdf
```

ddoc serve 띄워둔 상태라면 inline-envelope mode 로도 가능:

```bash
curl -s -X POST http://localhost:8765/report/render \
  -H 'Content-Type: application/json' \
  -d "{\"envelope\": $(cat /tmp/eda.json), \"format\": \"html\"}" \
  > /tmp/iris.html
```

## ddoc.yaml 컨벤션

각 generator 가 작성하는 dataset 디렉터리에는 ddoc plugin 이 modality
를 인식하기 위한 `ddoc.yaml` 이 함께 들어갑니다:

```
<dataset_dir>/
├── ddoc.yaml          ← modality 선언
├── data.csv           ← timeseries / text 의 경우
├── tone.wav           ← audio 의 경우
└── img_*.png          ← vision 의 경우
```

modality 별 `ddoc.yaml` 필드:

| modality | 필드 |
|---|---|
| `timeseries` | `modality: timeseries`, `csv_file: data.csv`, `timestamp_column: timestamp`, `numeric_columns: [x, y]` |
| `text` | `modality: text`, `text_column: text`, `id_column: id`, `language: english` |
| `audio` | `modality: audio`, `formats: [.wav]` |
| `image` | `modality: image`, `formats: [.png]` |
| `categorical` | `modality: categorical`, `distributions_file: distributions.json` (plugin 은 단일 평면 layout 만 읽음 — `<dataset_dir>/distributions.json` 직접) |

본인 데이터로 옮길 때는 같은 layout 을 따르면 됩니다 — `ddoc.yaml` 만
정확히 맞추면 plugin 이 자동으로 처리합니다.

## --scenario identical 의 용도

신규 deployment 가 ddoc 를 처음 켰을 때 "drift 가 진짜 0 이 나오는
환경인가?" 를 확인하는 sanity check:

```bash
ddoc examples generate timeseries --out /tmp/sane --scenario identical
ddoc analyze drift --data-path-ref /tmp/sane/ref --data-path-cur /tmp/sane/cur \
    --json --quiet
# → overall_score: 0.0
```

이 단계가 통과하면 plugin 설치 / 환경 변수 / DVC / cache 라우팅이 모두
정상이라는 의미. shifted scenario 로 옮기면 의미 있는 drift score 를
관찰할 수 있습니다.

## 다음 단계

- 저수준 분석 옵션: [데이터 분석](../guides/analysis.md)
- 다른 서비스에서 호출하기: [REST / GUI](serve.md)
- modality 별 plugin contract: 각 plugin 의 `README.md`
