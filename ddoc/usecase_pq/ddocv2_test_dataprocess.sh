#!/bin/bash

################################################################################
# ddoc Integration Test - Data Processing Pipeline
# 
# Usage: ./ddocv2_test_dataprocess.sh [reference_dataset] [target_dataset]
# Default: ./ddocv2_test_dataprocess.sh yolo_reference yolo_target
#
# 데이터 처리 파이프라인:
# 1. 환경 확인
# 2. 데이터셋 등록
# 3. 데이터 분석 (EDA)
# 4. 드리프트 분석 (reference vs target)
# 5. 결과 확인
################################################################################

set -e  # 에러 발생 시 스크립트 중단

# 인자 처리
REFERENCE_DATASET="${1:-yolo_reference}"
TARGET_DATASET="${2:-yolo_target}"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 프로젝트 루트 디렉토리
PROJECT_ROOT="/Users/bhc/dev/drift_v1/ddoc"
cd "$PROJECT_ROOT"

# 사용법 출력
print_usage() {
    echo "Usage: $0 [reference_dataset] [target_dataset]"
    echo ""
    echo "Arguments:"
    echo "  reference_dataset    Reference dataset directory name (default: yolo_reference)"
    echo "  target_dataset       Target dataset directory name (default: yolo_target)"
    echo ""
    echo "Example:"
    echo "  $0 yolo_reference yolo_target"
    echo "  $0 yolo_current yolo_target"
    echo ""
}

################################################################################
# 헬퍼 함수
################################################################################

print_header() {
    echo ""
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo ""
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_step() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 데이터셋 파일 개수 확인
count_files() {
    local dataset_path="$1"
    local count=0
    
    if [ -d "$dataset_path" ]; then
        # 이미지 파일 개수 확인
        count=$(find "$dataset_path" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
    fi
    
    echo "$count"
}

################################################################################
# 메인 실행
################################################################################

print_header "ddoc Data Processing Pipeline Test"
print_info "Dataset 1: $REFERENCE_DATASET"
print_info "Dataset 2: $TARGET_DATASET"
echo ""

# 가상환경 활성화
print_info "가상환경 활성화 중..."
source venv/bin/activate
print_step "가상환경 활성화 완료"

# 데이터셋 파일 개수 확인
REFERENCE_DATASET_COUNT=$(count_files "datasets/$REFERENCE_DATASET")
TARGET_DATASET_COUNT=$(count_files "datasets/$TARGET_DATASET")

print_info "$REFERENCE_DATASET: $REFERENCE_DATASET_COUNT 개 이미지 파일"
print_info "$TARGET_DATASET: $TARGET_DATASET_COUNT 개 이미지 파일"
echo ""

if [ "$REFERENCE_DATASET_COUNT" -eq 0 ] || [ "$TARGET_DATASET_COUNT" -eq 0 ]; then
    print_error "데이터셋 파일을 찾을 수 없습니다!"
    print_info "다음 디렉토리를 확인하세요:"
    echo "  - datasets/$REFERENCE_DATASET/"
    echo "  - datasets/$TARGET_DATASET/"
    exit 1
fi

################################################################################
# Phase 1: 데이터셋 등록
################################################################################

print_header "Phase 1: 데이터셋 등록"

# 기존 데이터셋 확인
print_info "기존 등록된 데이터셋 확인..."
ddoc dataset list
echo ""

# Dataset 1 등록
if ddoc dataset list | grep -q "$REFERENCE_DATASET"; then
    print_warning "$REFERENCE_DATASET이 이미 등록되어 있습니다."
else
    print_info "$REFERENCE_DATASET 등록 중... ($REFERENCE_DATASET_COUNT 개 이미지)"
    ddoc dataset add "$REFERENCE_DATASET" "datasets/$REFERENCE_DATASET"
    print_step "$REFERENCE_DATASET 등록 완료"
fi

# Dataset 2 등록
if ddoc dataset list | grep -q "$TARGET_DATASET"; then
    print_warning "$TARGET_DATASET가 이미 등록되어 있습니다."
else
    print_info "$TARGET_DATASET 등록 중... ($TARGET_DATASET_COUNT 개 이미지)"
    ddoc dataset add "$TARGET_DATASET" "datasets/$TARGET_DATASET"
    print_step "$TARGET_DATASET 등록 완료"
fi

# 등록 확인
echo ""
print_info "등록된 데이터셋 목록:"
ddoc dataset list
echo ""

################################################################################
# Phase 2: Dataset 1 분석
################################################################################

print_header "Phase 2: $REFERENCE_DATASET 분석 (EDA)"

print_info "$REFERENCE_DATASET 분석을 시작합니다..."
print_warning "이미지 파일만 자동으로 필터링됩니다 (.jpg, .png)"

# 분석 실행
ddoc analyze "$REFERENCE_DATASET"

print_step "$REFERENCE_DATASET 분석 완료"
echo ""

# 결과 확인
print_info "분석 결과 확인:"
echo "  - 디렉토리: analysis/$REFERENCE_DATASET/"
ls -la "analysis/$REFERENCE_DATASET/"
echo ""

################################################################################
# Phase 3: Dataset 2 분석
################################################################################

print_header "Phase 3: $TARGET_DATASET 분석 (EDA)"

print_info "$TARGET_DATASET 분석을 시작합니다..."
print_warning "이미지 파일만 자동으로 필터링됩니다 (.jpg, .png)"

# 분석 실행
ddoc analyze "$TARGET_DATASET"

print_step "$TARGET_DATASET 분석 완료"
echo ""

# 결과 확인
print_info "분석 결과 확인:"
echo "  - 디렉토리: analysis/$TARGET_DATASET/"
ls -la "analysis/$TARGET_DATASET/"
echo ""

################################################################################
# Phase 4: 드리프트 분석
################################################################################

print_header "Phase 4: 드리프트 분석 ($REFERENCE_DATASET vs $TARGET_DATASET)"

print_info "이종 데이터셋 드리프트 분석을 시작합니다..."
print_warning "Cross-dataset drift analysis: $REFERENCE_DATASET vs $TARGET_DATASET"

# 드리프트 분석 실행
ddoc drift-compare "$REFERENCE_DATASET" "$TARGET_DATASET" --output "analysis/drift_${REFERENCE_DATASET}_vs_${TARGET_DATASET}"

print_step "드리프트 분석 완료"
echo ""

# 결과 확인
print_info "드리프트 분석 결과 확인:"
echo "  - 디렉토리: analysis/drift_${REFERENCE_DATASET}_vs_${TARGET_DATASET}/"
ls -la "analysis/drift_${REFERENCE_DATASET}_vs_${TARGET_DATASET}/"
echo ""

################################################################################
# Phase 5: 결과 요약
################################################################################

print_header "Phase 5: 테스트 결과 요약"

echo "✅ 완료된 작업:"
echo "  1. ✓ 환경 확인"
echo "  2. ✓ 데이터셋 등록 ($REFERENCE_DATASET, $TARGET_DATASET)"
echo "  3. ✓ $REFERENCE_DATASET 분석 ($REFERENCE_DATASET_COUNT 개 이미지)"
echo "  4. ✓ $TARGET_DATASET 분석 ($TARGET_DATASET_COUNT 개 이미지)"
echo "  5. ✓ 드리프트 분석 ($REFERENCE_DATASET vs $TARGET_DATASET)"
echo ""

print_info "생성된 주요 디렉토리:"
echo "  - analysis/$REFERENCE_DATASET/                                    : $REFERENCE_DATASET 분석 결과"
echo "  - analysis/$TARGET_DATASET/                                    : $TARGET_DATASET 분석 결과"
echo "  - analysis/drift_${REFERENCE_DATASET}_vs_${TARGET_DATASET}/            : 드리프트 분석 결과"
echo ""

print_info "주요 파일:"
echo "  - analysis/$REFERENCE_DATASET/metrics.json                        : $REFERENCE_DATASET 분석 메트릭"
echo "  - analysis/$TARGET_DATASET/metrics.json                        : $TARGET_DATASET 분석 메트릭"
echo "  - analysis/drift_${REFERENCE_DATASET}_vs_${TARGET_DATASET}/metrics.json : 드리프트 메트릭"
echo "  - analysis/drift_${REFERENCE_DATASET}_vs_${TARGET_DATASET}/plots/images/ : 드리프트 시각화"
echo "  - datasets/$REFERENCE_DATASET/cache/                              : 캐시 파일"
echo "  - datasets/$TARGET_DATASET/cache/                              : 캐시 파일"
echo ""

print_step "🎉 데이터 처리 파이프라인 테스트 완료!"
echo ""

print_info "다음 명령어로 결과를 확인할 수 있습니다:"
echo "  ddoc dataset list                                        # 데이터셋 목록"
echo "  ddoc dataset history $REFERENCE_DATASET                          # 데이터셋 히스토리"
echo "  cat analysis/$REFERENCE_DATASET/metrics.json | python -m json.tool # 메트릭 확인"
echo "  cat analysis/drift_${REFERENCE_DATASET}_vs_${TARGET_DATASET}/metrics.json # 드리프트 확인"
echo "  open analysis/$REFERENCE_DATASET/plots/images/                    # 시각화 이미지 보기 (macOS)"
echo ""

################################################################################
# 종료
################################################################################

print_header "데이터 처리 테스트 스크립트 종료"
print_info "가상환경을 종료하려면: deactivate"
print_info "모델 학습을 진행하려면: ./ddocv2_test_modelprocess.sh $REFERENCE_DATASET $TARGET_DATASET"