"""
Audio Analysis Plugin Implementation for ddoc

Provides hookimpl for:
- eda_run: Audio attribute analysis
- drift_detect: Drift detection between baseline and current audio datasets
"""
import os
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

try:
    from ddoc.plugins.hookspecs import hookimpl
except ImportError:
    def hookimpl(func):
        return func

try:
    import librosa
except ImportError:
    librosa = None


class DOCAudioPlugin:
    """Audio Analysis Plugin for ddoc"""
    
    def _load_ddoc_yaml(self, dataset_path: Path) -> Dict[str, Any]:
        """Load and validate ddoc.yaml from dataset directory"""
        yaml_path = dataset_path / "ddoc.yaml"
        if not yaml_path.exists():
            raise ValueError(f"ddoc.yaml not found in {dataset_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config.get('modality') != 'audio':
            raise ValueError(f"Dataset {dataset_path} is not configured as audio modality")
        
        return config
    
    def _get_audio_files(self, directory: Path) -> list:
        """Get all audio files in directory"""
        audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
        audio_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(audio_extensions):
                    audio_files.append(Path(root) / file)
        return audio_files
    
    def _analyze_audio_attributes(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Calculate physical-based audio features"""
        if librosa is None:
            print("Warning: librosa not available")
            return None
        
        try:
            y, sr = librosa.load(str(file_path), sr=None, duration=30.0)  # Limit to 30s for speed
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = float(np.mean(zcr))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # MFCC statistics
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = [float(x) for x in np.mean(mfccs, axis=1)]
            mfcc_std = [float(x) for x in np.std(mfccs, axis=1)]
            
            return {
                'rms_energy_mean': rms_mean,
                'rms_energy_std': rms_std,
                'zcr_mean': zcr_mean,
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std
            }
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    @hookimpl
    def eda_run(self, snapshot_id, data_path, data_hash, output_path, invalidate_cache=False):
        """Run EDA for audio datasets"""
        from ddoc.core.cache_service import get_cache_service
        
        cache_service = get_cache_service()
        input_path = Path(data_path)
        output_path = Path(output_path)
        
        print(f"🚀 Audio EDA Analysis Started")
        print(f"=" * 80)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        metrics = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'snapshot_id': snapshot_id,
            'data_hash': data_hash,
            'modality': 'audio'
        }
        
        # Find audio datasets
        audio_datasets = []
        for item in input_path.iterdir():
            if item.is_dir():
                yaml_path = item / "ddoc.yaml"
                if yaml_path.exists():
                    try:
                        config = self._load_ddoc_yaml(item)
                        if config.get('modality') == 'audio':
                            audio_datasets.append((item, config))
                    except Exception as e:
                        print(f"⚠️ Skipping {item}: {e}")
        
        if not audio_datasets:
            print("⚠️ No audio datasets found")
            return None
        
        # Load cache
        attr_cache = {}
        if not invalidate_cache:
            attr_cache_data = cache_service.load_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="attributes_audio"
            )
            if attr_cache_data:
                attr_cache = attr_cache_data
        
        # Process each dataset
        all_attributes = {}
        
        for dataset_path, config in audio_datasets:
            print(f"\n📊 Processing dataset: {dataset_path.name}")
            
            audio_files = self._get_audio_files(dataset_path)
            print(f"   Found {len(audio_files)} audio files")
            
            for audio_file in audio_files:
                rel_path = str(audio_file.relative_to(input_path))
                attrs = self._analyze_audio_attributes(audio_file)
                if attrs:
                    all_attributes[rel_path] = attrs
        
        # Save cache
        if all_attributes:
            cache_service.save_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="attributes_audio",
                data=all_attributes
            )
        
        metrics['num_files'] = len(all_attributes)
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Audio Analysis Complete")
        
        return {
            "status": "success",
            "modality": "audio",
            "files_analyzed": len(all_attributes),
            "metrics_file": str(metrics_file),
            "summary": metrics
        }
    
    @hookimpl
    def drift_detect(
        self,
        snapshot_id_ref: str,
        snapshot_id_cur: str,
        data_path_ref: str,
        data_path_cur: str,
        data_hash_ref: str,
        data_hash_cur: str,
        detector: str,
        cfg: Dict[str, Any],
        output_path: str
    ) -> Optional[Dict[str, Any]]:
        """Detect drift between two audio snapshots"""
        from ddoc.core.cache_service import get_cache_service
        
        cache_service = get_cache_service()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load caches
        baseline_attr = cfg.get('baseline_cache') or cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_ref,
            data_hash=data_hash_ref,
            cache_type="attributes_audio"
        )
        
        current_attr = cfg.get('current_cache') or cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_cur,
            data_hash=data_hash_cur,
            cache_type="attributes_audio"
        )
        
        if not baseline_attr or not current_attr:
            return None
        
        drift_metrics = {
            'modality': 'audio',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        # Calculate drift for each metric
        metric_names = ['rms_energy_mean', 'zcr_mean', 'spectral_centroid_mean']
        drift_scores = []
        
        for metric in metric_names:
            ref_values = [a.get(metric, 0) for a in baseline_attr.values() if metric in a]
            cur_values = [a.get(metric, 0) for a in current_attr.values() if metric in a]
            
            if ref_values and cur_values:
                from scipy.stats import wasserstein_distance
                drift = wasserstein_distance(ref_values, cur_values)
                drift_scores.append(drift)
        
        drift_metrics['overall_score'] = float(np.mean(drift_scores)) if drift_scores else 0.0
        
        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(drift_metrics, f, indent=2)
        
        return drift_metrics

