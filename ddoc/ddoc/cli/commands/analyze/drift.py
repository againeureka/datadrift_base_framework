"""Drift analysis command"""
import typer
from rich import print
from typing import Optional

from ..utils import get_pmgr, _pretty
from ddoc.core.snapshot_service import get_snapshot_service
from ddoc.core.cache_service import get_cache_service


def analyze_drift_command(
    baseline: str = typer.Argument(..., help="Baseline snapshot ID or alias"),
    current: str = typer.Argument(..., help="Current snapshot ID or alias"),
    detector: str = typer.Option("mmd", "--detector", help="Drift detector method"),
):
    """
    Detect drift between two snapshots.
    
    필수적으로 두 스냅샷 간 비교를 수행합니다. 각 스냅샷의 분석 캐시를 사용하여
    효율적으로 drift를 계산합니다.
    
    Examples:
        ddoc analyze drift baseline v05        # baseline vs v05
        ddoc analyze drift v01 v02              # v01 vs v02
    """
    snapshot_service = get_snapshot_service()
    cache_service = get_cache_service()
    
    # Resolve snapshot IDs
    baseline_id = snapshot_service._resolve_version(baseline)
    current_id = snapshot_service._resolve_version(current)
    
    if not baseline_id:
        print(f"[red]❌ Snapshot '{baseline}' not found[/red]")
        return
    if not current_id:
        print(f"[red]❌ Snapshot '{current}' not found[/red]")
        return
    
    # Load snapshots
    snap_baseline = snapshot_service._load_snapshot(baseline_id)
    snap_current = snapshot_service._load_snapshot(current_id)
        
    if not snap_baseline:
        print(f"[red]❌ Failed to load snapshot {baseline_id}[/red]")
        return
    if not snap_current:
        print(f"[red]❌ Failed to load snapshot {current_id}[/red]")
        return
    
    # Load caches (attributes for drift detection)
    cache_baseline_attr = cache_service.load_analysis_cache(
        snapshot_id=baseline_id,
        data_hash=snap_baseline.data.dvc_hash,
        cache_type="attributes"
    )
    cache_current_attr = cache_service.load_analysis_cache(
        snapshot_id=current_id,
        data_hash=snap_current.data.dvc_hash,
        cache_type="attributes"
    )
    
    # Check if caches exist
    if not cache_baseline_attr:
        print(f"[yellow]⚠️  No analysis cache found for {baseline_id}[/yellow]")
        print(f"[yellow]   Run 'ddoc analyze eda {baseline_id}' first[/yellow]")
        return
    
    if not cache_current_attr:
        print(f"[yellow]⚠️  No analysis cache found for {current_id}[/yellow]")
        print(f"[yellow]   Run 'ddoc analyze eda {current_id}' first[/yellow]")
        return
    
    # Prepare configuration
    cfg = {
        "baseline_cache": cache_baseline_attr,
        "current_cache": cache_current_attr,
        "baseline_metadata": cache_service.load_file_metadata(
            snapshot_id=baseline_id,
            data_hash=snap_baseline.data.dvc_hash
        ),
        "current_metadata": cache_service.load_file_metadata(
            snapshot_id=current_id,
            data_hash=snap_current.data.dvc_hash
        )
    }
    
    # Output path
    output_path = f"analysis/drift_{baseline_id}_{current_id}"
    
    print(f"[cyan]🔍 Drift Analysis[/cyan]")
    print(f"   Baseline: {baseline_id} ({snap_baseline.data.dvc_hash[:7]})")
    print(f"   Current:  {current_id} ({snap_current.data.dvc_hash[:7]})")
    print()
    
    # Call plugins (multi-modal support: collect all non-None results)
    try:
        hook_results = get_pmgr().hook.drift_detect(
            snapshot_id_ref=baseline_id,
            snapshot_id_cur=current_id,
            data_path_ref=snap_baseline.data.path,
            data_path_cur=snap_current.data.path,
            data_hash_ref=snap_baseline.data.dvc_hash,
            data_hash_cur=snap_current.data.dvc_hash,
            detector=detector,
            cfg=cfg,
            output_path=output_path
        )
        
        # Collect all non-None results from all plugins
        if not hook_results:
            print("[red]❌ No plugin available for drift detection[/red]")
            print("[yellow]   Install plugins: pip install ddoc-full[/yellow]")
            return
        
        # Filter out None results
        valid_results = [r for r in hook_results if r is not None]
        
        if not valid_results:
            print("[red]❌ No plugin returned valid drift result[/red]")
            print("[yellow]   Install plugins: pip install ddoc-full[/yellow]")
            return
        
        # Merge results by modality
        merged_result = {
            "status": "success",
            "modalities": {},
            "summary": {}
        }
        
        pmgr = get_pmgr()
        for i, result in enumerate(valid_results):
            if isinstance(result, dict):
                # Try to identify modality from result or plugin name
                modality = result.get("modality")
                if not modality:
                    # Try to infer from plugin name (fallback)
                    try:
                        hook_impls = pmgr.pm.hook.drift_detect.get_hookimpls()
                        if i < len(hook_impls):
                            plugin_name = pmgr.pm.get_name(hook_impls[i].plugin)
                            if "vision" in plugin_name.lower():
                                modality = "image"
                            elif "text" in plugin_name.lower():
                                modality = "text"
                            elif "timeseries" in plugin_name.lower() or "ts" in plugin_name.lower():
                                modality = "timeseries"
                            elif "audio" in plugin_name.lower():
                                modality = "audio"
                            else:
                                modality = f"unknown_{i}"
                    except:
                        modality = f"unknown_{i}"
                
                merged_result["modalities"][modality] = result
                # Merge summary if available
                if "summary" in result:
                    merged_result["summary"][modality] = result["summary"]
        
        # For backward compatibility: if only one modality, use it as top-level result
        if len(merged_result["modalities"]) == 1:
            single_modality = list(merged_result["modalities"].keys())[0]
            res = merged_result["modalities"][single_modality]
        else:
            res = merged_result
        
        print(_pretty(res))
        
    except Exception as e:
        print(f"[red]❌ Error:[/red] {e}")
        import traceback
        traceback.print_exc()
