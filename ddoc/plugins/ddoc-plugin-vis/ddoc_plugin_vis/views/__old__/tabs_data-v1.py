from pathlib import Path
import streamlit as st

from core.constants import DEFAULT_DATA_ROOT
from services.dataset_manager import list_datasets, preview_samples
from services.dvc_cli import dvc_pull
from services.git_cli import git_checkout
from utils.eda import summarize_dataset, diff_stats
from utils.ui import metrics_row


def _dataset_select_box(project_root: Path, label: str):
    data_root = project_root / DEFAULT_DATA_ROOT
    options = [p.name for p in list_datasets(project_root)]
    if not options:
        st.info(f"{data_root} 아래에 데이터셋 폴더가 없습니다. 사이드바에서 업로드 하세요.")
        return None
    return st.selectbox(label, options)


def render_tab_data():
    project_root = Path(st.session_state.project_root)

    st.subheader("데이터셋 탐색 (단일)")
    one = _dataset_select_box(project_root, "데이터셋 선택")
    if one:
        ds_dir = project_root / DEFAULT_DATA_ROOT / one
        stats = summarize_dataset(ds_dir)
        metrics_row(stats)

        imgs = preview_samples(ds_dir, limit=12)
        if imgs:
            st.image([str(p) for p in imgs], width='stretch')
            
        #imgs = list(ds_dir.rglob("*.jpg"))[:12]
        #if imgs:
        #    st.caption("샘플 미리보기 (최대 12장)")
        #    st.image([str(p) for p in imgs])

    st.markdown("---")
    st.subheader("데이터셋 버전 비교 (2-way)")
    col1, col2 = st.columns(2)
    with col1:
        left_rev = st.text_input("좌측 리비전/브랜치/태그", value=st.session_state.compare_left)
    with col2:
        right_rev = st.text_input("우측 리비전/브랜치/태그", value=st.session_state.compare_right)

    if st.button("체크아웃 & dvc pull (좌→우 순서)"):
        if left_rev:
            out = git_checkout(left_rev, project_root)
            if out.returncode == 0:
                dvc_pull(project_root)
        if right_rev:
            out = git_checkout(right_rev, project_root)
            if out.returncode == 0:
                dvc_pull(project_root)
        st.success("체크아웃 및 데이터 동기화 완료 (마지막 상태 기준)")

    # 실제 비교는 동일한 루트 내 두 버전 폴더를 직접 가져오기 어려울 수 있으므로
    # 간단한 전략: 동일한 데이터셋 이름에 대해 스냅샷 디렉토리를 두 개 지정받아 비교
    left_path = st.text_input("좌 비교 경로(폴더)")
    right_path = st.text_input("우 비교 경로(폴더)")

    if st.button("폴더 기준 비교 실행"):
        lp = Path(left_path); rp = Path(right_path)
        if lp.exists() and rp.exists():
            ls = summarize_dataset(lp)
            rs = summarize_dataset(rp)
            metrics_row(ls, label_prefix="좌/")
            metrics_row(rs, label_prefix="우/")
            st.caption("차이 (우 - 좌)")
            diff = diff_stats(ls, rs)
            c1, c2, c3 = st.columns(3)
            c1.metric("Δ 이미지 수", int(diff.get('num_images', 0)))
            c2.metric("Δ 라벨 수", int(diff.get('num_labels_txt', 0)))
            c3.metric("Δ 용량(GB)", diff.get('size_gb', 0.0))
        else:
            st.error("비교 경로가 올바르지 않습니다.")