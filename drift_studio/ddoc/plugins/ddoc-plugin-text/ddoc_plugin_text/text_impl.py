"""
Text Analysis Plugin Implementation for ddoc

Provides hookimpl for:
- eda_run: Text attribute analysis, CLIP text embedding extraction
- drift_detect: Drift detection between baseline and current text datasets
"""
import os
import yaml
import json
import re
import zipfile
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

try:
    from ddoc.plugins.hookspecs import hookimpl
except ImportError:
    def hookimpl(func):
        return func

try:
    import torch
    import clip
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")


class DOCTextPlugin:
    """Text Analysis Plugin for ddoc"""
    
    def __init__(self):
        self.clip_model = None
        self.clip_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_clip_model(self):
        """Load CLIP model for text encoding"""
        if self.clip_model is None:
            print(f"Loading CLIP model (device: {self.device})...")
            self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
            self.clip_tokenizer = clip.tokenize
            print("CLIP model loaded")
    
    def _load_ddoc_yaml(self, dataset_path: Path) -> Dict[str, Any]:
        """Load and validate ddoc.yaml from dataset directory"""
        yaml_path = dataset_path / "ddoc.yaml"
        if not yaml_path.exists():
            raise ValueError(f"ddoc.yaml not found in {dataset_path}. Text datasets require ddoc.yaml with modality and schema definition.")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config.get('modality') != 'text':
            raise ValueError(f"Dataset {dataset_path} is not configured as text modality")
        
        # Validate required fields (only text_column is required now)
        if 'text_column' not in config:
            raise ValueError("ddoc.yaml must specify 'text_column'")
        
        return config
    
    def _find_csv_files(self, dataset_path: Path) -> Tuple[List[Path], Optional[Path]]:
        """
        Find CSV files in dataset directory.
        - If single CSV file exists, return it
        - If ZIP file exists, extract to temp directory and recursively find all CSV files
        - Otherwise, recursively find all CSV files in directory
        
        Returns:
            (csv_files, temp_extract_dir): List of CSV file paths and optional temp directory for ZIP extraction
        """
        csv_files = []
        temp_extract_dir = None
        
        # Check for ZIP files first
        zip_files = list(dataset_path.glob("*.zip"))
        if zip_files:
            # Extract ZIP and find CSV files recursively
            for zip_file in zip_files:
                print(f"   Found ZIP file: {zip_file.name}")
                # Create temporary directory for extraction (will be cleaned up later)
                temp_extract_dir = Path(tempfile.mkdtemp(prefix="ddoc_text_"))
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_extract_dir)
                        # Remove macOS metadata if present
                        macosx_dir = temp_extract_dir / "__MACOSX"
                        if macosx_dir.exists():
                            shutil.rmtree(macosx_dir)
                        
                        # Find all CSV files recursively in extracted directory
                        extracted_csvs = list(temp_extract_dir.rglob("*.csv"))
                        if extracted_csvs:
                            csv_files.extend(extracted_csvs)
                            print(f"   Found {len(extracted_csvs)} CSV files in ZIP")
                            for csv_path in extracted_csvs:
                                rel_path = csv_path.relative_to(temp_extract_dir)
                                print(f"      - {rel_path}")
                        else:
                            print(f"   ⚠️ No CSV files found in ZIP")
                except Exception as e:
                    print(f"   ⚠️ Error extracting {zip_file.name}: {e}")
                    if temp_extract_dir and temp_extract_dir.exists():
                        shutil.rmtree(temp_extract_dir)
                    temp_extract_dir = None
                    continue
        
        # If no ZIP files or no CSV found in ZIP, look for CSV files directly
        if not csv_files:
            # Check for single CSV file in root
            root_csvs = list(dataset_path.glob("*.csv"))
            if root_csvs:
                csv_files.extend(root_csvs)
                print(f"   Found {len(root_csvs)} CSV file(s) in root")
            else:
                # Recursively find all CSV files
                csv_files = list(dataset_path.rglob("*.csv"))
                if csv_files:
                    print(f"   Found {len(csv_files)} CSV file(s) recursively")
        
        return sorted(csv_files), temp_extract_dir
    
    def _load_and_combine_csvs(self, csv_files: List[Path], text_column: str, id_column: Optional[str] = None, base_path: Optional[Path] = None) -> pd.DataFrame:
        """Load and combine multiple CSV files into a single DataFrame"""
        dfs = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                if text_column not in df.columns:
                    # Try to get relative path for display
                    if base_path and csv_file.is_relative_to(base_path):
                        rel_path = csv_file.relative_to(base_path)
                    else:
                        rel_path = csv_file.name
                    print(f"   ⚠️ Text column '{text_column}' not found in {rel_path}")
                    continue
                
                # Add source file info for tracking
                if base_path and csv_file.is_relative_to(base_path):
                    rel_path = csv_file.relative_to(base_path)
                else:
                    rel_path = Path(csv_file.name)
                df['_source_file'] = str(rel_path)
                
                # Ensure ID column exists or create one
                if id_column and id_column in df.columns:
                    pass  # Use existing ID column
                elif id_column:
                    # ID column specified but not found, create sequential IDs
                    df[id_column] = [f"{rel_path.stem}_{i}" for i in range(len(df))]
                else:
                    # No ID column specified, try to find common ID columns
                    common_id_cols = ['id', 'ID', 'index', 'INDEX', 'idx']
                    found_id = None
                    for col in common_id_cols:
                        if col in df.columns:
                            found_id = col
                            break
                    if found_id:
                        id_column = found_id
                    else:
                        # Create sequential IDs
                        df['_auto_id'] = [f"{rel_path.stem}_{i}" for i in range(len(df))]
                        id_column = '_auto_id'
                
                dfs.append(df)
                print(f"   Loaded {len(df)} rows from {rel_path}")
                
            except Exception as e:
                print(f"   ⚠️ Error loading {csv_file}: {e}")
                continue
        
        if not dfs:
            return pd.DataFrame()
        
        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"   ✅ Total: {len(combined_df)} rows from {len(dfs)} files")
        
        return combined_df
    
    def _analyze_text_attributes(self, text: str, language: str = 'english') -> Dict[str, Any]:
        """Calculate physical-based text metrics"""
        if not text or pd.isna(text):
            return {
                'length_chars': 0,
                'length_words': 0,
                'whitespace_ratio': 0,
                'special_char_ratio': 0,
                'stopword_ratio': 0,
                'vocab_diversity': 0,
                'readability': 0
            }
        
        text_str = str(text)
        
        # Length metrics
        length_chars = len(text_str)
        words = word_tokenize(text_str.lower())
        length_words = len(words)
        
        # Whitespace ratio
        whitespace_count = sum(1 for c in text_str if c.isspace())
        whitespace_ratio = whitespace_count / length_chars if length_chars > 0 else 0
        
        # Special character ratio
        special_char_count = sum(1 for c in text_str if not c.isalnum() and not c.isspace())
        special_char_ratio = special_char_count / length_chars if length_chars > 0 else 0
        
        # Stopword ratio
        try:
            stop_words = set(stopwords.words(language))
            stopword_count = sum(1 for w in words if w in stop_words)
            stopword_ratio = stopword_count / length_words if length_words > 0 else 0
        except:
            stopword_ratio = 0
        
        # Vocabulary diversity (Type-Token Ratio)
        unique_words = len(set(words))
        vocab_diversity = unique_words / length_words if length_words > 0 else 0
        
        # Simple readability (Flesch-like approximation for English)
        readability = 0.0
        if language == 'english' and length_words > 0:
            sentences = re.split(r'[.!?]+', text_str)
            sentences = [s for s in sentences if s.strip()]
            avg_sentence_length = length_words / len(sentences) if sentences else 0
            # Simplified readability score (higher = easier)
            readability = max(0, min(100, 100 - (avg_sentence_length * 1.5)))
        
        return {
            'length_chars': length_chars,
            'length_words': length_words,
            'whitespace_ratio': whitespace_ratio,
            'special_char_ratio': special_char_ratio,
            'stopword_ratio': stopword_ratio,
            'vocab_diversity': vocab_diversity,
            'readability': readability
        }
    
    def _extract_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Extract CLIP text embedding"""
        if not text or pd.isna(text):
            return None
        
        self._load_clip_model()
        
        try:
            text_tokens = self.clip_tokenizer([str(text)], truncate=True).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                # Normalize
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                return text_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    @hookimpl
    def eda_run(self, snapshot_id, data_path, data_hash, output_path, invalidate_cache=False):
        """Run EDA for text datasets"""
        from ddoc.core.cache_service import get_cache_service
        from ddoc.core.schemas import FileMetadata
        
        cache_service = get_cache_service()
        input_path = Path(data_path)
        output_path = Path(output_path)
        
        print(f"🚀 Text EDA Analysis Started")
        print(f"=" * 80)
        print(f"Snapshot: {snapshot_id}")
        print(f"Data Hash: {data_hash[:8] if data_hash != 'unknown' else 'unknown'}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print()
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        metrics = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'snapshot_id': snapshot_id,
            'data_hash': data_hash,
            'modality': 'text'
        }
        
        # Find text datasets (directories with ddoc.yaml)
        text_datasets = []
        for item in input_path.iterdir():
            if item.is_dir():
                yaml_path = item / "ddoc.yaml"
                if yaml_path.exists():
                    try:
                        config = self._load_ddoc_yaml(item)
                        if config.get('modality') == 'text':
                            text_datasets.append((item, config))
                    except Exception as e:
                        print(f"⚠️ Skipping {item}: {e}")
        
        if not text_datasets:
            print("⚠️ No text datasets found (directories with ddoc.yaml and modality=text)")
            return None
        
        # Load caches
        attr_cache = {}
        emb_cache = {}
        
        if not invalidate_cache:
            attr_cache_data = cache_service.load_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="attributes_text"
            )
            if attr_cache_data:
                attr_cache = attr_cache_data
            
            emb_cache_data = cache_service.load_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="embedding_text"
            )
            if emb_cache_data:
                emb_cache = emb_cache_data
        
        # Process each text dataset
        all_attributes = {}
        all_embeddings = {}
        
        for dataset_path, config in text_datasets:
            print(f"\n📊 Processing dataset: {dataset_path.name}")
            print("-" * 80)
            
            text_column = config['text_column']
            id_column = config.get('id_column', None)
            language = config.get('language', 'english')
            
            # Find CSV files (handles single CSV, ZIP files, or recursive search)
            csv_files, temp_extract_dir = self._find_csv_files(dataset_path)
            
            if not csv_files:
                print(f"⚠️ No CSV files found in {dataset_path}")
                if temp_extract_dir and temp_extract_dir.exists():
                    shutil.rmtree(temp_extract_dir)
                continue
            
            # Determine base path for relative path calculation
            base_path = temp_extract_dir if temp_extract_dir else dataset_path
            
            # Load and combine all CSV files
            df = self._load_and_combine_csvs(csv_files, text_column, id_column, base_path)
            
            # Clean up temporary directory if created
            if temp_extract_dir and temp_extract_dir.exists():
                try:
                    shutil.rmtree(temp_extract_dir)
                except:
                    pass
            
            if df.empty:
                print(f"⚠️ No valid data loaded from {dataset_path}")
                continue
            
            # Determine actual ID column to use
            actual_id_column = id_column
            if not actual_id_column:
                # Try to find common ID columns
                common_id_cols = ['id', 'ID', 'index', 'INDEX', 'idx', '_auto_id']
                for col in common_id_cols:
                    if col in df.columns:
                        actual_id_column = col
                        break
            
            # Analyze attributes
            print(f"   Analyzing {len(df)} texts...")
            for idx, row in df.iterrows():
                text = row[text_column]
                
                # Generate row ID
                if actual_id_column and actual_id_column in df.columns:
                    row_id = str(row[actual_id_column])
                else:
                    row_id = f"row_{idx}"
                
                # Include source file in cache key for better tracking
                source_file = row.get('_source_file', dataset_path.name)
                cache_key = f"{dataset_path.name}/{source_file}/{row_id}"
                
                # Attributes
                attrs = self._analyze_text_attributes(text, language)
                all_attributes[cache_key] = attrs
                
                # Embedding
                embedding = self._extract_text_embedding(text)
                if embedding is not None:
                    all_embeddings[cache_key] = {
                        'embedding': embedding.tolist(),
                        'text_length': len(str(text))
                    }
            
            print(f"   ✅ Analyzed {len(df)} texts")
        
        # Save caches
        if all_attributes:
            cache_service.save_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="attributes_text",
                data=all_attributes
            )
            print(f"💾 Saved {len(all_attributes)} attribute records")
        
        if all_embeddings:
            cache_service.save_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="embedding_text",
                data=all_embeddings
            )
            print(f"💾 Saved {len(all_embeddings)} embedding records")
        
        # Calculate summary statistics
        if all_attributes:
            metrics['num_texts'] = len(all_attributes)
            metrics['avg_length_chars'] = np.mean([a['length_chars'] for a in all_attributes.values()])
            metrics['avg_length_words'] = np.mean([a['length_words'] for a in all_attributes.values()])
            metrics['avg_vocab_diversity'] = np.mean([a['vocab_diversity'] for a in all_attributes.values()])
        
        # Save metrics
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Text Analysis Complete")
        print(f"   📄 Metrics: {metrics_file}")
        
        return {
            "status": "success",
            "modality": "text",
            "texts_analyzed": metrics.get('num_texts', 0),
            "metrics_file": str(metrics_file),
            "output_path": str(output_path),
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
        """Detect drift between two text snapshots"""
        from ddoc.core.cache_service import get_cache_service
        
        cache_service = get_cache_service()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 Text Drift Detection Started")
        print(f"=" * 80)
        
        # Load caches
        baseline_attr = cfg.get('baseline_cache') or cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_ref,
            data_hash=data_hash_ref,
            cache_type="attributes_text"
        )
        baseline_emb = cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_ref,
            data_hash=data_hash_ref,
            cache_type="embedding_text"
        )
        
        current_attr = cfg.get('current_cache') or cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_cur,
            data_hash=data_hash_cur,
            cache_type="attributes_text"
        )
        current_emb = cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_cur,
            data_hash=data_hash_cur,
            cache_type="embedding_text"
        )
        
        if not baseline_attr or not current_attr:
            print("❌ Missing baseline or current data")
            return None
        
        drift_metrics = {
            'modality': 'text',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        # Attribute drift (PSI for each metric)
        print("📈 Attribute Drift:")
        print("-" * 80)
        
        attribute_names = ['length_chars', 'length_words', 'whitespace_ratio', 
                          'special_char_ratio', 'stopword_ratio', 'vocab_diversity', 'readability']
        
        attribute_drifts = {}
        for attr_name in attribute_names:
            ref_values = [a.get(attr_name, 0) for a in baseline_attr.values()]
            cur_values = [a.get(attr_name, 0) for a in current_attr.values()]
            
            if ref_values and cur_values:
                try:
                    # Use PSI for drift
                    from scipy.stats import wasserstein_distance
                    psi = wasserstein_distance(ref_values, cur_values) / (np.mean(ref_values) + 1e-10)
                    attribute_drifts[attr_name] = float(psi)
                    print(f"   {attr_name:20s} Drift: {psi:.4f}")
                except:
                    attribute_drifts[attr_name] = 0.0
        
        drift_metrics['attribute_drifts'] = attribute_drifts
        drift_metrics['attribute_drift_overall'] = np.mean(list(attribute_drifts.values())) if attribute_drifts else 0.0
        
        # Embedding drift
        if baseline_emb and current_emb:
            print("\n🧠 Embedding Drift:")
            print("-" * 80)
            
            ref_emb_list = [np.array(e['embedding']) for e in baseline_emb.values()]
            cur_emb_list = [np.array(e['embedding']) for e in current_emb.values()]
            
            if ref_emb_list and cur_emb_list:
                ref_emb_array = np.array(ref_emb_list)
                cur_emb_array = np.array(cur_emb_list)
                
                # Cosine distance between mean embeddings
                ref_mean = ref_emb_array.mean(axis=0)
                cur_mean = cur_emb_array.mean(axis=0)
                cosine_sim = np.dot(ref_mean, cur_mean) / (np.linalg.norm(ref_mean) * np.linalg.norm(cur_mean) + 1e-10)
                embedding_drift = 1.0 - cosine_sim
                
                drift_metrics['embedding_drift'] = float(embedding_drift)
                print(f"   Embedding Drift (cosine): {embedding_drift:.4f}")
            else:
                drift_metrics['embedding_drift'] = 0.0
        
        # Overall score
        attr_score = drift_metrics.get('attribute_drift_overall', 0)
        emb_score = drift_metrics.get('embedding_drift', 0)
        overall = 0.5 * attr_score + 0.5 * emb_score
        drift_metrics['overall_score'] = float(overall)
        
        # Save metrics
        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(drift_metrics, f, indent=2)
        
        print(f"\n✅ Text Drift Detection Complete")
        print(f"   📄 Metrics: {metrics_file}")
        
        return drift_metrics

