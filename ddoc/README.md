# ddoc - Data Drift Detection & Analysis Framework

> Git-like workflow for MLOps with snapshot-based version management

**ddoc**은 데이터, 코드, 실험을 통합적으로 관리하는 MLOps 도구입니다. Git과 유사한 직관적인 워크플로우로 머신러닝 프로젝트의 완벽한 재현성을 보장합니다.

## ✨ 주요 기능

- 📦 **Workspace Management**: 자동 프로젝트 스캐폴딩
- 📸 **Snapshot System**: Git-like 버전 관리 (데이터 + 코드 + 실험)
- 🔬 **Data Analysis**: EDA 및 Drift 감지
- 🧪 **Experiment Tracking**: Trainer 기반 실험 시스템
- 🔌 **Plugin Architecture**: 확장 가능한 플러그인 시스템

## 🚀 빠른 시작

### 설치

```bash
pip install ddoc
```

### 5분 튜토리얼

```bash
# 1. 프로젝트 초기화
ddoc init myproject
cd myproject

# 2. 데이터 추가
ddoc add --data ./datasets/train_data

# 3. 첫 스냅샷 생성
git add . && git commit -m "Initial setup"
ddoc snapshot -m "baseline" -a baseline

# 4. 데이터 분석
ddoc analyze eda

# 5. 실험 실행
ddoc exp train yolo --dataset train_data
```

더 자세한 튜토리얼은 [시작하기 가이드](docs/tutorial/quick-start.md)를 참조하세요.

## 📚 문서

### 시작하기
- **[설치 가이드](docs/tutorial/installation.md)** - 설치 및 요구사항
- **[빠른 시작](docs/tutorial/quick-start.md)** - 5분 튜토리얼
- **[핵심 개념](docs/tutorial/concepts.md)** - Workspace, Snapshot, Alias 이해하기

### 사용자 가이드
- **[워크스페이스 관리](docs/guides/workspace.md)** - 프로젝트 초기화 및 파일 관리
- **[스냅샷 관리](docs/guides/snapshots.md)** - 버전 관리 및 복원
- **[Trainer 시스템](docs/guides/trainer.md)** - Trainer 기반 실험 시스템
- **[데이터 분석](docs/guides/analysis.md)** - EDA 및 Drift 감지
- **[실험 관리](docs/guides/experiments.md)** - 실험 실행 및 추적

### 레퍼런스
- **[명령어 레퍼런스](docs/reference/commands.md)** - 모든 명령어 상세 설명

### 고급 사용법
- **[워크플로우](docs/advanced/workflows.md)** - 고급 워크플로우 및 베스트 프랙티스
- **[문제 해결](docs/advanced/troubleshooting.md)** - 자주 발생하는 문제 해결

### 마이그레이션
- **[v1.x → v2.0 마이그레이션](docs/migration/v1-to-v2.md)** - v1.x에서 업그레이드

전체 문서는 [docs/](docs/) 디렉토리에서 확인하세요.

## 📦 버전

- **v2.0.3** (Current) - [릴리스 노트](docs/releases/v2.0.3.md)
- **v2.0.2** - [릴리스 노트](docs/releases/v2.0.2.md)
- **v2.0.1** - [릴리스 노트](docs/releases/v2.0.1.md)
- **v2.0.0** - [릴리스 노트](docs/releases/v2.0.0.md)
- **v1.3.6** (Legacy) - [릴리스 노트](docs/releases/v1.3.6.md)

[전체 변경 이력](docs/changelog.md) | [릴리스 노트](docs/releases/)

## 🎯 주요 사용 사례

### 데이터 버전 관리
```bash
ddoc init myproject
ddoc add --data ./datasets/train_data
ddoc snapshot -m "baseline dataset" -a baseline
```

### 실험 추적
```bash
ddoc exp train yolo --dataset train_data --model yolov8n.pt
ddoc exp best train_data  # 최고 성능 실험 찾기
```

### 데이터 드리프트 감지
```bash
ddoc analyze drift baseline production
```

## 🤝 기여

기여를 환영합니다! 기여 가이드는 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

## 📄 라이선스

MIT License

## 👥 기여자

- JPark @ KETI
- Ethicsense @ KETI

---