"""services/dataset_manager.py
A robust, extensible data manager for mixed-modality datasets and common CV formats.
- Safe file scanning (avoids PIL.UnidentifiedImageError by verifying images before preview)
- Nested directory support (any depth via rglob)
- Modality-aware stats (image, text, audio, video, csv/timeseries, other)
- Dataset format hints (YOLO, Pascal VOC; easily extendable)
- Upload helpers (save/extract/track/commit) kept for convenience
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import streamlit as st

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

from core.constants import DEFAULT_DATA_ROOT, UPLOADS_DIR
from services.dvc_cli import dvc_add, dvc_push
from services.git_cli import git_add, git_commit, git_tag, git_push, git_branch_create


# -------------------------------
# Enums / Typing
# -------------------------------

class FileType(Enum):
    IMAGE = auto()
    TEXT = auto()
    AUDIO = auto()
    VIDEO = auto()
    CSV = auto()         # timeseries/tabular
    XML = auto()
    JSON = auto()
    OTHER = auto()


class DatasetFormat(Enum):
    UNKNOWN = auto()
    YOLO = auto()        # e.g., images/* and labels/*.txt (class cx cy w h)
    PASCAL_VOC = auto()  # Annotations/*.xml + JPEGImages/*.jpg
    COCO = auto()        # images/*.jpg + annotations/*.json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
TEXT_EXTS  = {".txt", ".md", ".log"}
CSV_EXTS   = {".csv"}
XML_EXTS   = {".xml"}
JSON_EXTS  = {".json"}


# -------------------------------
# Upload helpers (unchanged API)
# -------------------------------

def save_uploaded_zip(upload, project_root: Path) -> Path:
    updir = project_root / UPLOADS_DIR
    updir.mkdir(exist_ok=True, parents=True)
    fpath = updir / upload.name
    with open(fpath, 'wb') as f:
        f.write(upload.getbuffer())
    return fpath


def extract_zip(zip_path: Path, target_dir: Path) -> None:
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


def ensure_data_branch(dataset_name: str, project_root: Path, prefix: str = "data/") -> str:
    branch = prefix + dataset_name
    git_branch_create(branch, checkout=True, project_root=project_root)
    return branch


# -------------------------------
# Core scanning & validation
# -------------------------------

def _classify(path: Path) -> FileType:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return FileType.IMAGE
    if ext in AUDIO_EXTS:
        return FileType.AUDIO
    if ext in VIDEO_EXTS:
        return FileType.VIDEO
    if ext in CSV_EXTS:
        return FileType.CSV
    if ext in XML_EXTS:
        return FileType.XML
    if ext in JSON_EXTS:
        return FileType.JSON
    if ext in TEXT_EXTS:
        return FileType.TEXT
    return FileType.OTHER


def _iter_files(root: Path) -> Iterable[Path]:
    # rglob handles arbitrary nesting: works for d3/images/*.jpg and d3/d3/images/*.jpg
    return (p for p in root.rglob("*") if p.is_file())


def _is_valid_image(path: Path) -> bool:
    """Verify images safely to avoid PIL.UnidentifiedImageError during preview.
    Only attempts verification for known image extensions. Corrupt or mislabeled
    files are filtered out.
    """
    if _classify(path) is not FileType.IMAGE:
        return False
    if not PIL_AVAILABLE:
        # If PIL is unavailable, fall back to trusting the extension
        return True
    try:
        with Image.open(path) as im:
            im.verify()  # lightweight header check
        return True
    except Exception:
        return False


def enumerate_valid_images(root: Path, limit: Optional[int] = None) -> List[Path]:
    valid: List[Path] = []
    for p in _iter_files(root):
        if _is_valid_image(p):
            valid.append(p)
            if limit is not None and len(valid) >= limit:
                break
    return valid


@dataclass
class DatasetStats:
    num_images: int = 0
    num_audio: int = 0
    num_video: int = 0
    num_text: int = 0
    num_csv: int = 0
    num_xml: int = 0
    num_json: int = 0
    num_other: int = 0
    size_gb: float = 0.0
    format_hint: DatasetFormat = DatasetFormat.UNKNOWN


def scan_stats(folder: Path) -> DatasetStats:
    total_bytes = 0
    counts = {t: 0 for t in FileType}

    for p in _iter_files(folder):
        try:
            total_bytes += p.stat().st_size
        except Exception:
            pass
        counts[_classify(p)] += 1

    fmt = _detect_format(folder)

    return DatasetStats(
        num_images=counts[FileType.IMAGE],
        num_audio=counts[FileType.AUDIO],
        num_video=counts[FileType.VIDEO],
        num_text=counts[FileType.TEXT],
        num_csv=counts[FileType.CSV],
        num_xml=counts[FileType.XML],
        num_json=counts[FileType.JSON],
        num_other=counts[FileType.OTHER],
        size_gb=round(total_bytes / (1024 ** 3), 3),
        format_hint=fmt,
    )


# -------------------------------
# Format detectors (heuristics)
# -------------------------------

def _detect_format(root: Path) -> DatasetFormat:
    """Heuristics to guess dataset format. Non-fatal, best-effort."""
    # YOLO v5/8 style: labels/*.txt and images/*.{jpg,png,...}
    labels = list(root.rglob("labels/*.txt"))
    images = [p for p in root.rglob("images/*") if p.suffix.lower() in IMAGE_EXTS]
    if labels and images:
        return DatasetFormat.YOLO

    # Pascal VOC: Annotations/*.xml with JPEGImages/*.jpg (common layout)
    if list(root.rglob("Annotations/*.xml")) and list(root.rglob("JPEGImages/*")):
        return DatasetFormat.PASCAL_VOC

    # COCO: annotations/*.json + images/*
    if list(root.rglob("annotations/*.json")) and images:
        return DatasetFormat.COCO

    return DatasetFormat.UNKNOWN


# -------------------------------
# Public API for dataset browsing
# -------------------------------

def summarize_dataset(ds_dir: Path) -> Dict[str, float | str]:
    stats = scan_stats(ds_dir)
    return {
        "num_images": float(stats.num_images),
        "num_audio": float(stats.num_audio),
        "num_video": float(stats.num_video),
        "num_text": float(stats.num_text),
        "num_csv": float(stats.num_csv),
        "num_xml": float(stats.num_xml),
        "num_json": float(stats.num_json),
        "num_other": float(stats.num_other),
        "size_gb": stats.size_gb,
        "format": stats.format_hint.name,
    }


def list_datasets(project_root: Path) -> List[Path]:
    data_root = project_root / DEFAULT_DATA_ROOT
    if not data_root.exists():
        return []
    return [p for p in data_root.glob("*") if p.is_dir()]


def preview_samples(ds_dir: Path, limit: int = 12) -> List[Path]:
    """Return a small set of *verified* image paths for safe UI preview.
    This prevents PIL.UnidentifiedImageError by skipping corrupt/mislabeled files.
    """
    return enumerate_valid_images(ds_dir, limit=limit)


# -------------------------------
# Simple comparisons for EDA
# -------------------------------

def diff_stats(left: Dict[str, float | str], right: Dict[str, float | str]) -> Dict[str, float]:
    keys = {k for k, v in left.items() if isinstance(v, (int, float))} | {k for k, v in right.items() if isinstance(v, (int, float))}
    out: Dict[str, float] = {}
    for k in sorted(keys):
        out[k] = float((right.get(k, 0.0) or 0.0) - (left.get(k, 0.0) or 0.0))
    return out


# -------------------------------
# Optional helpers for format-specific parsing (stubs for extensibility)
# -------------------------------

def parse_yolo_label_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    """Returns (cls, cx, cy, w, h) if parseable; else None."""
    try:
        parts = line.strip().split()
        if len(parts) != 5:
            return None
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:])
        return (cls, cx, cy, w, h)
    except Exception:
        return None


def count_yolo_labels(labels_root: Path) -> int:
    cnt = 0
    for txt in labels_root.rglob("*.txt"):
        try:
            if txt.stat().st_size == 0:
                continue
            with open(txt, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if parse_yolo_label_line(line) is not None:
                        cnt += 1
        except Exception:
            continue
    return cnt


# -------------------------------
# Notes for integrators
# -------------------------------
# - Use `summarize_dataset()` for dashboard metrics.
# - Use `preview_samples()` to render images safely in Streamlit: st.image([str(p) for p in preview_samples(...)])
# - `scan_stats()` now counts multiple modalities; extend EXT sets as needed.
# - `_detect_format()` provides a hint to display format-specific UI (YOLO/VOC/COCO).
# - Future: add lightweight audio/video probing (duration) via ffprobe if available.
