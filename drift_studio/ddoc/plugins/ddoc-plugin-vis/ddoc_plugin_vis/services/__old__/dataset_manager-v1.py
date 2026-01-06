import zipfile
from pathlib import Path
from typing import Dict, List
import streamlit as st

from core.constants import DEFAULT_DATA_ROOT, UPLOADS_DIR, DATA_BRANCH_PREFIX
from .dvc_cli import dvc_add, dvc_push
from .git_cli import git_add, git_commit, git_tag, git_push, git_checkout, git_branch_create


def save_uploaded_zip(upload, project_root: Path) -> Path:
    updir = project_root / UPLOADS_DIR
    updir.mkdir(exist_ok=True)
    fpath = updir / upload.name
    with open(fpath, 'wb') as f:
        f.write(upload.getbuffer())
    return fpath


def extract_zip(zip_path: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(target_dir)


def track_and_commit_dataset(dataset_dir: Path, project_root: Path, message: str, tag: str = "", push_remote: bool = True):
    dvc_add(dataset_dir, project_root)
    git_add([str(dataset_dir) + ".dvc", ".gitignore"], project_root)
    git_commit(message, project_root)
    if tag:
        git_tag(tag, project_root)
    if push_remote:
        dvc_push(project_root)
        git_push(bool(tag), project_root)


def scan_stats(folder: Path) -> Dict[str, float]:
    images = list(folder.rglob("*.jpg")) + list(folder.rglob("*.jpeg")) + list(folder.rglob("*.png"))
    labels = list(folder.rglob("*.txt"))
    size_bytes = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    return {
        "num_images": float(len(images)),
        "num_labels_txt": float(len(labels)),
        "size_gb": round(size_bytes / (1024**3), 3)
    }


def list_datasets(project_root: Path) -> List[Path]:
    data_root = project_root / DEFAULT_DATA_ROOT
    return [p for p in data_root.glob("*") if p.is_dir()]


def ensure_data_branch(dataset_name: str, project_root: Path):
    branch = DATA_BRANCH_PREFIX + dataset_name
    git_branch_create(branch, checkout=True, project_root=project_root)
    return branch
