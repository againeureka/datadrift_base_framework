"""EDA analysis command"""
import typer
from rich import print
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..utils import get_pmgr, _pretty
from ddoc.core.snapshot_service import get_snapshot_service
from ddoc.core.cache_service import get_cache_service


def analyze_eda_command(
    snapshot: Optional[str] = typer.Argument(None, help="Snapshot ID or alias (default: current workspace)"),
    invalidate_cache: bool = typer.Option(False, "--invalidate-cache", help="Invalidate existing cache before analysis"),
    save_snapshot: bool = typer.Option(False, "--save-snapshot", help="Save workspace as permanent snapshot after analysis"),
):
    """
    Run EDA analysis on a snapshot or current workspace.
    
    스냅샷을 지정하지 않으면 현재 워크스페이스 상태를 분석합니다.
    --save-snapshot 옵션으로 분석 후 영구 스냅샷으로 저장할 수 있습니다.
    
    Examples:
        ddoc analyze eda                    # 현재 워크스페이스 분석
        ddoc analyze eda --save-snapshot     # 현재 워크스페이스 + 영구 저장
        ddoc analyze eda baseline           # baseline 스냅샷 분석
        ddoc analyze eda v01                 # v01 스냅샷 분석
    """
    snapshot_service = get_snapshot_service()
    cache_service = get_cache_service()
    
    # Resolve snapshot or use workspace state
    if snapshot:
        snapshot_id = snapshot_service._resolve_version(snapshot)
        if not snapshot_id:
            print(f"[red]❌ Snapshot '{snapshot}' not found[/red]")
            return
        
        snap = snapshot_service._load_snapshot(snapshot_id)
        if not snap:
            print(f"[red]❌ Failed to load snapshot {snapshot_id}[/red]")
            return
        
        # Get snapshot data info
        snapshot_data_hash = snap.data.dvc_hash
        # Snapshot data path is always relative to project root: "data/"
        # Actual path is project_root / "data/"
        snapshot_data_path = str(snapshot_service.project_root / snap.data.path)
        
        # Check current workspace data hash
        current_data_hash = snapshot_service._get_dvc_data_hash() or "unknown"
        
        # Always use current workspace data path for analysis
        # (Analysis operates on actual files in workspace)
        data_path = "data/"  # Relative path from project root
        
        # Use snapshot data hash for cache lookup
        # (Cache is keyed by data hash, so same data = same cache)
        data_hash = snapshot_data_hash
        
        # Warn if data hashes don't match (snapshot data not in workspace)
        if snapshot_data_hash != current_data_hash and current_data_hash != "unknown":
            print(f"[yellow]⚠️  Snapshot data hash ({snapshot_data_hash[:8]}) differs from current workspace ({current_data_hash[:8]})[/yellow]")
            print(f"[yellow]   Analyzing current workspace data, but using snapshot cache if available[/yellow]")
            print(f"[yellow]   To analyze snapshot data, run: ddoc snapshot --restore {snapshot_id}[/yellow]\n")
        
        is_workspace = False
    else:
        # Use workspace state
        # Ensure DVC tracking is up to date before analysis
        # This guarantees consistent hash between EDA and snapshot
        data_changed = snapshot_service._has_data_changes()
        if data_changed:
            print("[cyan]📦 Updating DVC tracking to ensure consistent hash...[/cyan]")
            import subprocess
            result = subprocess.run(
                ["dvc", "add", "data/"],
                cwd=snapshot_service.project_root,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"[yellow]⚠️  DVC add failed: {result.stderr}[/yellow]")
                print("[yellow]   Analysis will proceed but hash may be inconsistent.[/yellow]")
            else:
                print("[green]✅ DVC tracking updated[/green]")
        
        workspace_state = snapshot_service.get_workspace_state()
        snapshot_id = workspace_state["snapshot_id"]
        data_path = workspace_state["data"]["path"]
        data_hash = workspace_state["data"]["dvc_hash"]
        is_workspace = True
        
        print(f"[cyan]📊 Current workspace data_hash: {data_hash[:8]}...[/cyan]")
        
        if data_hash == "unknown":
            print("[yellow]⚠️  No DVC tracking found. Run 'ddoc add --data' first.[/yellow]")
            print("[yellow]   Analysis will proceed but cache may not be reusable.[/yellow]")
        else:
            # Sync workspace cache with current data hash for incremental analysis
            sync_result = cache_service.sync_workspace_cache(data_hash)
            if sync_result.get("synced"):
                print(f"[cyan]🔄 Synced cache: {sync_result['old_hash']} → {sync_result['new_hash']}[/cyan]")
                print(f"[cyan]   Cache types: {', '.join(sync_result['cache_types'])}[/cyan]")
    
    # Check cache
    if not invalidate_cache:
        # Try to find cache by data hash
        cache = cache_service.load_analysis_cache(data_hash=data_hash, cache_type="summary")
        if cache:
            print(f"[cyan]📋 Found existing cache for data hash {data_hash[:8]}...[/cyan]")
            # Check if there are other snapshots with same data hash
            snapshots_with_same_data = cache_service.find_snapshots_by_data_hash(data_hash)
            if len(snapshots_with_same_data) > 0:
                print(f"[cyan]   Shared with snapshots: {', '.join(snapshots_with_same_data)}[/cyan]")
    
    # Output path
    if is_workspace:
        output_path = "analysis/workspace"
        print(f"[cyan]📊 Analyzing current workspace state[/cyan]")
        if not save_snapshot:
            print(f"[yellow]💡 Tip: Use --save-snapshot to save this state as a permanent snapshot[/yellow]\n")
    else:
        output_path = f"analysis/{snapshot_id}"
        print(f"[cyan]📊 Analyzing snapshot: {snapshot_id}[/cyan]\n")
    
    # Call plugins (multi-modal support: collect all non-None results)
    try:
        hook_results = get_pmgr().hook.eda_run(
            snapshot_id=snapshot_id,
            data_path=data_path,
            data_hash=data_hash,
            output_path=output_path,
            invalidate_cache=invalidate_cache
        )
        
        # Collect all non-None results from all plugins
        # pluggy returns a list of results from all hook implementations
        if not hook_results:
            print("[red]❌ No plugin available for EDA analysis[/red]")
            print("[yellow]   Install plugins: pip install ddoc-full[/yellow]")
            return
        
        # Filter out None results and collect all plugin results
        valid_results = [r for r in hook_results if r is not None]
        
        if not valid_results:
            print("[red]❌ No plugin returned valid result[/red]")
            print("[yellow]   Install plugins: pip install ddoc-full[/yellow]")
            return
        
        # Merge results by modality
        # Each plugin should return a dict with 'modality' key or we infer from plugin name
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
                        hook_impls = pmgr.pm.hook.eda_run.get_hookimpls()
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
        
        # Verify hash consistency after analysis (for workspace analysis)
        if is_workspace and isinstance(res, dict) and res.get("status") == "success":
            # Re-check hash to detect changes during analysis
            final_data_hash = snapshot_service._get_dvc_data_hash() or "unknown"
            if final_data_hash != data_hash and data_hash != "unknown":
                print(f"[yellow]⚠️  Data hash changed during analysis![/yellow]")
                print(f"   Before: {data_hash[:8]}...")
                print(f"   After:  {final_data_hash[:8]}...")
                print(f"[yellow]   Data may have been modified during analysis.[/yellow]")
                print(f"[yellow]   Cache was saved with old hash. Consider re-running analysis.[/yellow]")
                # Update data_hash for snapshot creation
                data_hash = final_data_hash
        
        # Save snapshot if requested
        if is_workspace and save_snapshot and isinstance(res, dict) and res.get("status") == "success":
            message = input("\nEnter snapshot message (or press Enter for default): ").strip()
            if not message:
                message = f"EDA analysis: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            alias_input = input("Enter snapshot alias (optional, press Enter to skip): ").strip()
            alias = alias_input if alias_input else None
            
            result = snapshot_service.create_snapshot(
                message=message,
                alias=alias,
                auto_commit=True
            )
            
            if result["success"]:
                # Migrate cache to new snapshot
                new_snapshot_id = result["snapshot_id"]
                cache_service._save_snapshot_mapping(new_snapshot_id, data_hash)
                print(f"[green]✅ Saved as snapshot: {new_snapshot_id}[/green]")
                if alias:
                    print(f"[green]   Alias: {alias}[/green]")
        
    except Exception as e:
        print(f"[red]❌ Error:[/red] {e}")
        import traceback
        traceback.print_exc()
