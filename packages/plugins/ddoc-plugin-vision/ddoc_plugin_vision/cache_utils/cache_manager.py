import os
import json
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from .cache_repository import CacheRepository

'''
dataset_folder/
├── cache/
│   ├── analysis_image_analysis_test_data2.cache
│   ├── analysis_image_analysis_test_data2_meta.json
│   ├── analysis_image_drift_content_test_data2.cache
│   ├── analysis_image_drift_content_test_data2_meta.json
│   ├── analysis_xai_analysis_test_data2.cache
│   └── analysis_xai_analysis_test_data2_meta.json
'''
DEFAULT_CACHE_DATA_TYPES: Tuple[str, ...] = (
    "attribute_analysis",
    "embedding_analysis",
)


class CacheManager:
    def __init__(self, dataset_directory):
        """캐시 매니저 초기화 (데이터셋별 캐시만 지원)"""
        if not dataset_directory:
            raise ValueError("dataset_directory는 필수입니다. 전역 캐시는 지원하지 않습니다.")
        
        # 데이터셋별 캐시 디렉토리. 저장할 때만 만든다: 캐시를 찾기만 하는
        # 비교(ddoc diff)는 사용자의 데이터 폴더에 흔적을 남기지 않는다.
        self.cache_dir = Path(dataset_directory) / 'cache'
        
        # 캐시 설정
        self.cache_expiry_days = int(os.getenv('DATADRIFT_CACHE_EXPIRY_DAYS', '30'))
        self.max_cache_size_mb = int(os.getenv('DATADRIFT_CACHE_MAX_SIZE_MB', '999999'))
    
    def _get_cache_key(self, identifier, content_type="html"):
        """캐시 키 생성"""
        # 식별자를 안전한 파일명으로 변환
        safe_identifier = self._make_safe_filename(identifier)
        
        # 파일명 길이 제한 (너무 길어지지 않도록)
        if len(safe_identifier) > 50:
            # 해시를 사용하여 짧게 만들기
            hash_obj = hashlib.md5(identifier.encode())
            safe_identifier = f"{safe_identifier[:30]}_{hash_obj.hexdigest()[:8]}"

        prefix = f"{content_type}_" if content_type else ""
        return f"{prefix}{safe_identifier}"
    
    def _make_safe_filename(self, filename):
        """파일명을 안전하게 만듭니다."""
        # 특수문자 제거 및 공백을 언더스코어로 변경
        import re
        safe_name = re.sub(r'[^\w\-_.]', '_', filename)
        # 연속된 언더스코어를 하나로 변경
        safe_name = re.sub(r'_+', '_', safe_name)
        # 앞뒤 언더스코어 제거
        safe_name = safe_name.strip('_')
        return safe_name
    
    def _get_cache_path(self, cache_key):
        """캐시 파일 경로 반환"""
        return self.cache_dir / f"{cache_key}.cache"
    
    def _get_metadata_path(self, cache_key):
        """메타데이터 파일 경로 반환"""
        return self.cache_dir / f"{cache_key}_meta.json"
    
    def _is_cache_valid(self, metadata_path):
        """캐시가 유효한지 확인"""
        if not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # 만료 시간 확인
            created_time = datetime.fromisoformat(metadata['created_time'])
            expiry_time = created_time + timedelta(days=self.cache_expiry_days)
            
            return datetime.now() < expiry_time
        except Exception:
            return False
    
    def _save_metadata(self, metadata_path, content_size, content_type="html"):
        """메타데이터 저장"""
        metadata = {
            'created_time': datetime.now().isoformat(),
            'content_size': content_size,
            'content_type': content_type,
            'expiry_days': self.cache_expiry_days
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _cleanup_expired_cache(self):
        """만료된 캐시 정리"""
        for cache_file in self.cache_dir.glob("*.cache"):
            metadata_path = cache_file.with_name(f"{cache_file.stem}_meta.json")
            if not self._is_cache_valid(metadata_path):
                try:
                    cache_file.unlink()
                    if metadata_path.exists():
                        metadata_path.unlink()
                except Exception as e:
                    print(f"캐시 정리 중 오류: {e}")
    
    def _check_cache_size(self):
        """캐시 크기 확인 및 정리"""
        total_size = 0
        cache_files = []
        
        for cache_file in self.cache_dir.glob("*.cache"):
            size = cache_file.stat().st_size
            total_size += size
            cache_files.append((cache_file, size))
        
        # MB로 변환
        total_size_mb = total_size / (1024 * 1024)
        
        if total_size_mb > self.max_cache_size_mb:
            print(f"⚠️  캐시 크기 초과: {total_size_mb:.2f}MB > {self.max_cache_size_mb}MB")
            print(f"📁 캐시 파일들:")
            for cache_file, size in cache_files:
                try:
                    metadata_path = cache_file.with_name(f"{cache_file.stem}_meta.json")
                    cache_file.unlink()
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    total_size_mb -= size / (1024 * 1024)
                    if total_size_mb <= self.max_cache_size_mb * 0.8:  # 80%까지 정리
                        break
                except Exception as e:
                    print(f"캐시 크기 정리 중 오류: {e}")
    
    def get_cached_content(self, identifier, content_type="html"):
        """캐시된 컨텐츠 가져오기"""
        cache_key = self._get_cache_key(identifier, content_type)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)
        
        # 캐시가 존재하고 유효한지 확인
        if cache_path.exists() and self._is_cache_valid(metadata_path):
            try:
                with open(cache_path, 'rb') as f:
                    content = pickle.load(f)
                return content
            except Exception as e:
                print(f"캐시 읽기 오류: {e}")
                return None
        
        return None
    
    def save_cached_content(self, identifier, content, content_type="html"):
        """컨텐츠를 캐시에 저장"""
        cache_key = self._get_cache_key(identifier, content_type)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)
        
        try:
            print(f"💾 캐시 저장 중: {cache_key}")
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            
            # 컨텐츠를 pickle로 저장
            with open(cache_path, 'wb') as f:
                pickle.dump(content, f)
            
            # 저장된 파일 크기 확인
            file_size = cache_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            print(f"📁 저장된 파일 크기: {file_size_mb:.2f}MB")
            
            # 메타데이터 저장
            content_size = len(str(content).encode('utf-8'))
            self._save_metadata(metadata_path, content_size, content_type)
            
            # 캐시 크기 확인 및 정리 (저장 후에만)
            self._check_cache_size()
            
            return True
        except Exception as e:
            print(f"❌ 캐시 저장 오류: {e}")
            return False
    
    def invalidate_cache(self, identifier, content_type="html"):
        """특정 캐시 무효화"""
        cache_key = self._get_cache_key(identifier, content_type)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            return True
        except Exception as e:
            print(f"캐시 무효화 오류: {e}")
            return False
    
    def clear_all_cache(self):
        """모든 캐시 정리"""
        try:
            for cache_file in self.cache_dir.glob("*"):
                cache_file.unlink()
            return True
        except Exception as e:
            print(f"전체 캐시 정리 오류: {e}")
            return False
    
    def get_cache_info(self):
        """캐시 정보 반환"""
        cache_files = list(self.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size / (1024 * 1024)
        
        # 캐시 파일 상세 정보
        cache_details = []
        for cache_file in cache_files:
            metadata_path = cache_file.with_name(f"{cache_file.stem}_meta.json")
            file_info = {
                'filename': cache_file.name,
                'size_mb': round(cache_file.stat().st_size / (1024 * 1024), 3),
                'created_time': None,
                'content_type': None
            }
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    file_info['created_time'] = metadata.get('created_time')
                    file_info['content_type'] = metadata.get('content_type')
                except Exception:
                    pass
            
            cache_details.append(file_info)
        
        return {
            'total_files': len(cache_files),
            'total_size_mb': round(total_size_mb, 2),
            'max_size_mb': self.max_cache_size_mb,
            'expiry_days': self.cache_expiry_days,
            'cache_dir': str(self.cache_dir.absolute()),
            'cache_details': cache_details
        }

def get_cache_manager(dataset_directory):
    """데이터셋별 캐시 매니저 반환 (dataset_directory 필수)"""
    if not dataset_directory:
        raise ValueError("dataset_directory는 필수입니다. 전역 캐시는 지원하지 않습니다.")
    return CacheManager(dataset_directory)


def get_cache_repository(dataset_directory, dataset_name: Optional[str] = None) -> CacheRepository:
    """중앙 저장소 캐시 매니저 반환"""

    return CacheRepository.from_directory(dataset_directory, dataset_name=dataset_name)

def get_cached_html_content(identifier, generator_func, *args, dataset_directory):
    """HTML 컨텐츠를 캐시에서 가져오거나 생성"""
    cache_manager = get_cache_manager(dataset_directory)
    
    # 캐시에서 확인
    cached_content = cache_manager.get_cached_content(identifier, "html")
    if cached_content is not None:
        return cached_content
    
    # 캐시에 없으면 생성
    try:
        content = generator_func(*args)
        cache_manager.save_cached_content(identifier, content, "html")
        return content
    except Exception as e:
        # 에러가 발생하면 캐시에 저장하지 않고 예외를 다시 발생시킴
        raise e


def _parse_created_time(meta_path: Path) -> datetime:
    try:
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        return datetime.fromisoformat(metadata.get('created_time'))
    except Exception:
        # fallback to file mtime
        try:
            return datetime.fromtimestamp(meta_path.stat().st_mtime)
        except Exception:
            return datetime.min


def get_latest_cached_content_by_prefix(directory, identifier_prefix: str, exclude_stem: str | None = None):
    """가장 최근 메타 기준으로 identifier_prefix에 해당하는 캐시 내용을 로드

    Args:
        directory: 데이터셋 디렉토리
        identifier_prefix: 캐시 식별자 접두어 (예: "attribute_analysis_test_data_v")
        exclude_stem: 제외할 캐시 stem (예: 현재 버전의 정확한 stem)
    Returns:
        Tuple[content, stem] 또는 (None, None)
    """
    cache_manager = get_cache_manager(directory)
    candidates = []
    for cache_file in cache_manager.cache_dir.glob("*.cache"):
        stem = cache_file.stem
        if not stem.startswith(identifier_prefix):
            continue
        if exclude_stem and stem == exclude_stem:
            continue
        meta_path = cache_file.with_name(f"{stem}_meta.json")
        created = _parse_created_time(meta_path)
        candidates.append((created, cache_file, meta_path, stem))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, cache_file, _, stem = candidates[0]
    try:
        with open(cache_file, 'rb') as f:
            content = pickle.load(f)
        return content, stem
    except Exception:
        return None, None


def _build_identifier(dir_name: str, data_type: str, version: Optional[str] = None) -> str:
    return f"{data_type}_{dir_name}_{version}" if version else f"{data_type}_{dir_name}"

def _validate_cache_integrity(directory, cached_data, data_type):
    """캐시 데이터의 무결성을 검증"""
    directory = Path(directory)
    
    # 현재 실제 파일 목록 수집
    formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    current_files = set()
    
    # 직접 파일들
    for fmt in formats:
        current_files.update(f.name for f in directory.glob(f"*{fmt}"))
    
    # YOLO 구조 파일들
    yolo_subdirs = ['train/images', 'valid/images', 'test/images']
    for subdir in yolo_subdirs:
        subdir_path = directory / subdir
        if subdir_path.exists():
            for fmt in formats:
                current_files.update(f.name for f in subdir_path.glob(f"*{fmt}"))
    
    # 캐시된 파일 목록
    cached_files = set(cached_data.keys())
    
    # 파일 목록 비교
    if current_files != cached_files:
        print(f"📊 File mismatch detected:")
        print(f"   Current files: {len(current_files)}")
        print(f"   Cached files: {len(cached_files)}")
        if current_files - cached_files:
            print(f"   Added: {len(current_files - cached_files)} files")
        if cached_files - current_files:
            print(f"   Removed: {len(cached_files - current_files)} files")
        return False
    
    # 내용 변경 감지 (가능한 경우에 한함)
    try:
        # 파일 크기/mtime 기반 변경 감지: attribute_analysis는 size(MB)를 보관함
        if data_type == "attribute_analysis":
            for fname in cached_files:
                cached_entry = cached_data.get(fname, {})
                cached_size_mb = cached_entry.get('size')
                # 실제 파일 경로 탐색 (flat + YOLO)
                file_path = None
                candidates = [directory / fname,
                              directory / 'train/images' / fname,
                              directory / 'valid/images' / fname,
                              directory / 'test/images' / fname]
                for c in candidates:
                    if c.exists():
                        file_path = c
                        break
                if not file_path:
                    continue
                real_size_mb = file_path.stat().st_size / (1024 * 1024)
                if cached_size_mb is not None and abs(real_size_mb - float(cached_size_mb)) > 0.01:
                    print(f"🔄 Detected content change in '{fname}' (size diff). Invalidating cache.")
                    return False
        # embedding_analysis는 파일 목록만 보관 → 내용 변경은 상위 로직에서 처리
    except Exception:
        # 보수적으로 통과
        pass

    return True

def get_cached_analysis_data(directory, data_type="image_analysis"):
    """분석 데이터를 캐시에서 가져오거나 생성 (무결성 검증 포함)"""
    cache_manager = get_cache_manager(directory)
    
    identifier = _build_identifier(os.path.basename(directory), data_type)
    
    # 캐시에서 확인 (content_type을 빈 문자열로 하여 접두사 없이 로드)
    cached_data = cache_manager.get_cached_content(identifier, "")
    if cached_data is not None:
        # 데이터 무결성 검증
        if _validate_cache_integrity(directory, cached_data, data_type):
            return cached_data
        else:
            print(f"⚠️ Cache validation failed for {identifier}, invalidating...")
            cache_manager.invalidate_cache(identifier, "")
            return None
    
    return None

def get_cached_analysis_data_by_version(directory, data_type="image_analysis", version=None):
    """버전 정보를 포함한 캐시 조회 (무결성 검증 포함)"""
    cache_manager = get_cache_manager(directory)
    
    identifier = _build_identifier(os.path.basename(directory), data_type, version)

    cached_data = cache_manager.get_cached_content(identifier, "")
    
    if cached_data is not None:
        # 데이터 무결성 검증
        if _validate_cache_integrity(directory, cached_data, data_type):
            return cached_data
        else:
            print(f"⚠️ Cache validation failed for {identifier}, invalidating...")
            cache_manager.invalidate_cache(identifier, "")
            return None
    
    return None

def save_analysis_data(directory, data, data_type="image_analysis"):
    """분석 데이터를 캐시에 저장"""
    cache_manager = get_cache_manager(directory)

    identifier = _build_identifier(os.path.basename(directory), data_type)

    # content_type을 빈 문자열로 하여 접두사 없이 저장
    return cache_manager.save_cached_content(identifier, data, "")

def save_analysis_data_by_version(directory, data, data_type="image_analysis", version=None):
    """버전 정보를 포함한 캐시 저장"""
    cache_manager = get_cache_manager(directory)

    identifier = _build_identifier(os.path.basename(directory), data_type, version)

    return cache_manager.save_cached_content(identifier, data, "")


def export_local_cache_to_repository(
    dataset_directory: str | Path,
    version: Optional[str],
    data_types: Iterable[str] = DEFAULT_CACHE_DATA_TYPES,
    dataset_name: Optional[str] = None,
) -> Dict[str, Path]:
    """현재 로컬 캐시를 중앙 저장소로 내보냅니다."""

    cache_manager = get_cache_manager(dataset_directory)
    repository = get_cache_repository(dataset_directory, dataset_name=dataset_name)
    # Use directory basename for identifier (matches how cache was saved)
    dir_name = os.path.basename(dataset_directory)
    saved_paths: Dict[str, Path] = {}

    for data_type in data_types:
        identifier = _build_identifier(dir_name, data_type, version)
        content = cache_manager.get_cached_content(identifier, "")
        if content is None:
            continue
        saved_paths[data_type] = repository.save(version or "unknown", data_type, content)

    return saved_paths


def import_repository_cache_to_local(
    dataset_directory: str | Path,
    version: Optional[str],
    data_types: Iterable[str] = DEFAULT_CACHE_DATA_TYPES,
    *,
    clear_existing: bool = False,
    dataset_name: Optional[str] = None,
) -> Dict[str, bool]:
    """중앙 저장소 캐시를 로컬 캐시 디렉토리에 복원합니다."""

    cache_manager = get_cache_manager(dataset_directory)
    repository = get_cache_repository(dataset_directory, dataset_name=dataset_name)

    if clear_existing:
        cache_manager.clear_all_cache()

    # Use directory basename for identifier (matches how cache is saved)
    dir_name = os.path.basename(dataset_directory)
    restored: Dict[str, bool] = {}

    for data_type in data_types:
        content = repository.load(version or "unknown", data_type)
        if content is None:
            restored[data_type] = False
            continue
        identifier = _build_identifier(dir_name, data_type, version)
        cache_manager.save_cached_content(identifier, content, "")
        restored[data_type] = True

    return restored


def repository_has_cache(
    dataset_directory: str | Path, version: Optional[str], data_type: str
) -> bool:
    repository = get_cache_repository(dataset_directory)
    cache_path = repository.version_dir(version or "unknown") / f"{data_type}.cache"
    return cache_path.exists()

 