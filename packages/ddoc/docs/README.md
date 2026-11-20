# ddoc Documentation

## 시작하기

- **[설치 가이드](tutorial/installation.md)** — 설치와 요구사항
- **[빠른 시작](tutorial/quick-start.md)** — 5분 만에 첫 drift 판정
- **[핵심 개념](tutorial/concepts.md)** — 판정, modality, 플러그인, envelope

## 튜토리얼

- **[toy 데이터](tutorial/toy-data.md)** — modality 별 예제 데이터와 dataset layout
- **[detector cookbook](tutorial/detectors.md)** — `--detector` 선택과 severity 임계값
- **[레시피](tutorial/recipe.md)** — 여러 단계를 YAML 하나로
- **[REST / GUI](tutorial/serve.md)** — `ddoc serve`
- **[배포](tutorial/deploy.md)** — 컨테이너로 운영하기

## 가이드

- **[데이터 분석](guides/analysis.md)** — `ddoc analyze drift|eda`

## 레퍼런스

- **[치트시트](cheatsheet.md)** — 상황별로 그대로 칠 명령 (`ddoc scenarios`, `./run.sh menu`)
- **[명령어 레퍼런스](reference/commands.md)** — 모든 CLI 명령과 옵션

## 릴리스 정보

- **[Changelog](changelog.md)** — 버전별 변경사항 요약
- **[릴리스 노트](releases/)** — 각 버전의 상세 릴리스 노트

## 문서 고치기

표 일부는 코드에서 만들어집니다. `<!-- generated:… -->` 표시 사이는 손으로 고치지 말고
코드(예: `ddoc/archetypes.py` 의 판단 기준)를 고친 뒤 다시 만드세요.

```bash
./run.sh docs build    # 생성 부분 다시 만들기 (유형 판독표, 품질 소견표)
./run.sh docs check    # 최신인지 확인 — ./run.sh test 와 CI 가 함께 확인
./run.sh docs shots    # README 스크린샷 다시 찍기 (Chrome 필요)
```

명령 레퍼런스의 옵션 설명은 손으로 쓰지만, 각 명령의 모든 옵션이 적혀 있는지는 테스트가
확인합니다(`tests/test_docs.py`).
