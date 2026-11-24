# ddoc-full

ddoc with all analysis plugins (vision, text, timeseries, audio)

## 설치 방법

### PyPI 에서 (배포 후)

```bash
pip install ddoc-full
```

ddoc 과 플러그인은 아직 PyPI 에 올라가 있지 않아 지금은 이 명령이 동작하지 않습니다. 그때까지는 아래처럼
각 패키지를 폴더 경로로 설치하거나, 저장소 루트에서 `./run.sh setup all` 을 쓰세요.

### 개발 환경 설치 (로컬)

로컬 개발 환경에서는 각 패키지를 개발 모드로 설치해야 합니다:

```bash
# 방법 1: 설치 스크립트 사용 (권장)
cd packages/plugins/ddoc-full
./install_dev.sh

# 방법 2: 수동 설치
pip install -e ../../ddoc  # ddoc core
pip install -e ../ddoc-plugin-vision
pip install -e ../ddoc-plugin-text
pip install -e ../ddoc-plugin-timeseries
pip install -e ../ddoc-plugin-audio
pip install -e ../ddoc-plugin-categorical
pip install -e ../ddoc-plugin-tabular
```

### 설치 확인

```bash
# 플러그인 목록 확인
ddoc plugin list

# 버전 확인
ddoc --version
```

## 포함된 플러그인

- `ddoc-plugin-vision`: 이미지 분석
- `ddoc-plugin-text`: 텍스트 분석
- `ddoc-plugin-timeseries`: 시계열 분석
- `ddoc-plugin-audio`: 오디오 분석

## 버전 관리

`ddoc-full`의 버전은 `ddoc` 코어와 동일한 메이저/마이너 버전을 사용합니다:
- `ddoc 2.1.0` = `ddoc-full 2.1.0`

플러그인들은 독립적으로 버전 관리되며, 최소 버전 요구사항으로 명시됩니다.

