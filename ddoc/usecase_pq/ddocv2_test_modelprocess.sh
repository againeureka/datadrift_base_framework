#!/bin/bash

################################################################################
# ddoc Integration Test - Model Processing Pipeline
# 
# Usage: ./ddocv2_test_modelprocess.sh [reference_dataset] [current_dataset]
# Default: ./ddocv2_test_modelprocess.sh yolo_reference yolo_current
#
# 모델 처리 파이프라인:
# 1. 환경 확인
# 2. 실험 설정 확인
# 3. 모델 학습 (Reference)
# 4. 모델 학습 (Current)
# 5. 실험 결과 확인
################################################################################

set -e  # 에러 발생 시 스크립트 중단

# 인자 처리
REFERENCE_DATASET="${1:-yolo_reference}"
CURRENT_DATASET="${2:-yolo_current}"

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
    echo "Usage: $0 [reference_dataset] [current_dataset]"
    echo ""
    echo "Arguments:"
    echo "  reference_dataset    Reference dataset directory name (default: yolo_reference)"
    echo "  current_dataset      Current dataset directory name (default: yolo_current)"
    echo ""
    echo "Example:"
    echo "  $0 yolo_reference yolo_current"
    echo "  $0 my_ref_data my_cur_data"
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

print_header "ddoc Model Processing Pipeline Test"
print_info "Dataset 1: $REFERENCE_DATASET"
print_info "Dataset 2: $CURRENT_DATASET"
echo ""

# 가상환경 활성화
print_info "가상환경 활성화 중..."
source venv/bin/activate
print_step "가상환경 활성화 완료"

# 데이터셋 파일 개수 확인
REFERENCE_DATASET_COUNT=$(count_files "datasets/$REFERENCE_DATASET")
CURRENT_DATASET_COUNT=$(count_files "datasets/$CURRENT_DATASET")

print_info "$REFERENCE_DATASET: $REFERENCE_DATASET_COUNT 개 이미지 파일"
print_info "$CURRENT_DATASET: $CURRENT_DATASET_COUNT 개 이미지 파일"
echo ""

if [ "$REFERENCE_DATASET_COUNT" -eq 0 ] || [ "$CURRENT_DATASET_COUNT" -eq 0 ]; then
    print_error "데이터셋 파일을 찾을 수 없습니다!"
    print_info "다음 디렉토리를 확인하세요:"
    echo "  - datasets/$REFERENCE_DATASET/"
    echo "  - datasets/$CURRENT_DATASET/"
    exit 1
fi

################################################################################
# Phase 1: 실험 설정 확인
################################################################################

print_header "Phase 1: 실험 설정 확인"

# params.yaml 확인
print_info "실험 설정 확인 중..."
if [ -f "params.yaml" ]; then
    print_step "params.yaml 파일 확인됨"
    echo ""
    print_info "실험 설정:"
    grep -A 10 "experiments:" params.yaml || print_warning "experiments 섹션을 찾을 수 없습니다"
    echo ""
else
    print_error "params.yaml 파일을 찾을 수 없습니다!"
    exit 1
fi

# 기존 실험 결과 확인
print_info "기존 실험 결과 확인..."
if [ -d "experiments/" ]; then
    print_warning "기존 실험 결과가 있습니다:"
    ls -la experiments/
    echo ""
    print_info "기존 실험을 정리하시겠습니까? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_info "기존 실험 결과 정리 중..."
        rm -rf experiments/
        print_step "기존 실험 결과 정리 완료"
    fi
else
    print_info "새로운 실험을 시작합니다"
fi

echo ""

################################################################################
# Phase 2: Reference 모델 학습
################################################################################

print_header "Phase 2: Reference 모델 학습 ($REFERENCE_DATASET)"

print_info "$REFERENCE_DATASET 데이터셋으로 Reference 모델 학습을 시작합니다..."
print_warning "학습 시간이 오래 걸릴 수 있습니다 (epoch 2 설정)"

# Reference 모델 학습
print_info "Reference 모델 학습 중... (exp_ref)"
ddoc train "$REFERENCE_DATASET" --epochs 2 --batch 8 --device cpu --name exp_ref

print_step "Reference 모델 학습 완료"
echo ""

# 결과 확인
print_info "Reference 모델 학습 결과 확인:"
echo "  - 디렉토리: experiments/exp_ref/"
if [ -d "experiments/exp_ref/" ]; then
    ls -la "experiments/exp_ref/"
    echo ""
    print_info "학습 메트릭:"
    if [ -f "experiments/exp_ref/experiment_metadata.json" ]; then
        cat experiments/exp_ref/experiment_metadata.json | python -m json.tool
    fi
else
    print_warning "Reference 모델 학습 결과를 찾을 수 없습니다"
fi
echo ""

################################################################################
# Phase 3: Current 모델 학습
################################################################################

print_header "Phase 3: Current 모델 학습 ($CURRENT_DATASET)"

print_info "$CURRENT_DATASET 데이터셋으로 Current 모델 학습을 시작합니다..."
print_warning "학습 시간이 오래 걸릴 수 있습니다 (epoch 2 설정)"

# Current 모델 학습
print_info "Current 모델 학습 중... (exp_cur)"
ddoc train "$CURRENT_DATASET" --epochs 2 --batch 8 --device cpu --name exp_cur

print_step "Current 모델 학습 완료"
echo ""

# 결과 확인
print_info "Current 모델 학습 결과 확인:"
echo "  - 디렉토리: experiments/exp_cur/"
if [ -d "experiments/exp_cur/" ]; then
    ls -la "experiments/exp_cur/"
    echo ""
    print_info "학습 메트릭:"
    if [ -f "experiments/exp_cur/experiment_metadata.json" ]; then
        cat experiments/exp_cur/experiment_metadata.json | python -m json.tool
    fi
else
    print_warning "Current 모델 학습 결과를 찾을 수 없습니다"
fi
echo ""

################################################################################
# Phase 4: 실험 결과 확인
################################################################################

print_header "Phase 4: 실험 결과 확인"

# 실험 목록 확인
print_info "실험 목록 확인:"
ddoc exp list
echo ""

# 실험 상세 정보 확인
print_info "Reference 실험 상세 정보:"
ddoc exp show exp_ref
echo ""

print_info "Current 실험 상세 정보:"
ddoc exp show exp_cur
echo ""

# 실험 비교
print_info "실험 비교:"
ddoc exp compare exp_ref exp_cur
echo ""

################################################################################
# Phase 5: 결과 요약
################################################################################

print_header "Phase 5: 테스트 결과 요약"

echo "✅ 완료된 작업:"
echo "  1. ✓ 환경 확인"
echo "  2. ✓ 실험 설정 확인"
echo "  3. ✓ Reference 모델 학습 ($REFERENCE_DATASET)"
echo "  4. ✓ Current 모델 학습 ($CURRENT_DATASET)"
echo "  5. ✓ 실험 결과 확인"
echo ""

print_info "생성된 주요 디렉토리:"
echo "  - experiments/exp_ref/                    : Reference 모델 학습 결과"
echo "  - experiments/exp_cur/                     : Current 모델 학습 결과"
echo ""

print_info "주요 파일:"
echo "  - experiments/exp_ref/weights/best.pt     : Reference 모델 (최고 성능)"
echo "  - experiments/exp_cur/weights/best.pt     : Current 모델 (최고 성능)"
echo "  - experiments/exp_ref/experiment_metadata.json : Reference 실험 메타데이터"
echo "  - experiments/exp_cur/experiment_metadata.json : Current 실험 메타데이터"
echo "  - experiments/exp_ref/train_ref.log       : Reference 학습 로그"
echo "  - experiments/exp_cur/train_cur.log       : Current 학습 로그"
echo ""

print_step "🎉 모델 처리 파이프라인 테스트 완료!"
echo ""

print_info "다음 명령어로 결과를 확인할 수 있습니다:"
echo "  ddoc exp list                             # 실험 목록"
echo "  ddoc exp show exp_ref                    # Reference 실험 상세"
echo "  ddoc exp show exp_cur                    # Current 실험 상세"
echo "  ddoc exp compare exp_ref exp_cur          # 실험 비교"
echo "  cat experiments/exp_ref/experiment_metadata.json | python -m json.tool"
echo "  cat experiments/exp_cur/experiment_metadata.json | python -m json.tool"
echo ""

################################################################################
# 종료
################################################################################

print_header "모델 처리 테스트 스크립트 종료"
print_info "가상환경을 종료하려면: deactivate"
print_info "전체 파이프라인을 다시 실행하려면: ./ddocv2_test_dataprocess.sh $REFERENCE_DATASET $CURRENT_DATASET"
