"""Plugin management commands"""
from typing import Optional
import typer
from rich import print
from .utils import get_pmgr, _pretty


def plugin_list_command():
    """
    List all installed plugins (without loading heavy dependencies).
    
    This command uses entry point metadata only, avoiding expensive imports
    like PyTorch and scikit-learn for fast execution.
    """
    print("[bold cyan]🔌 Installed Plugins:[/bold cyan]")

    # Use importlib.metadata to read entry points without loading plugins
    import importlib.metadata
    
    try:
        entry_points = importlib.metadata.entry_points(group="ddoc")
    except Exception as e:
        print(f"[red]❌ Failed to read plugins: {e}[/red]")
        return
    
    # Convert to list for compatibility with older Python versions
    plugins = list(entry_points) if hasattr(entry_points, '__iter__') else entry_points
    
    if plugins:
        print(f"\n[bold]Found {len(plugins)} plugins:[/bold]")
        print("-" * 60)
        print(f"{'Name':<20} {'Type':<15} {'Status':<10}")
        print("-" * 60)
        
        for ep in plugins:
            name = ep.name
            # Try to get plugin type from name
            if 'vision' in name.lower():
                plugin_type = "vision"
            elif 'yolo' in name.lower():
                plugin_type = "object-detect"
            elif 'nlp' in name.lower():
                plugin_type = "nlp"
            elif 'core' in name.lower() or 'builtin' in name.lower():
                plugin_type = "core"
            else:
                plugin_type = "extension"
            
            status = "installed"
            
            print(f"{name:<20} {plugin_type:<15} {status:<10}")
        
        print("-" * 60)
        print("\n[dim]💡 Tip: Use 'ddoc plugin info <name>' for detailed information[/dim]")
    else:
        print("  No plugins installed.")


def plugin_info_command(
    plugin_name: Optional[str] = typer.Argument(None, help="Specific plugin name to show info for")
):
    """
    Show detailed information about plugins.
    
    Examples:
        ddoc plugin info
        ddoc plugin info ddoc_vision
    """
    print("[bold magenta]🔍 Plugin Information[/bold magenta]")
    
    # Get metadata from all plugins
    metadata_list = get_pmgr().hook.ddoc_get_metadata()
    
    if not metadata_list:
        print("[yellow]No plugins provided metadata.[/yellow]")
        return
    
    if plugin_name:
        # Show specific plugin info
        plugin_metadata = next(
            (meta for meta in metadata_list if meta.get('name') == plugin_name),
            None
        )
        
        if plugin_metadata:
            print(f"\n[bold]Plugin: {plugin_name}[/bold]")
            print(_pretty(plugin_metadata))
        else:
            print(f"[red]❌ Plugin '{plugin_name}' not found.[/red]")
    else:
        # Show all plugins metadata
        print(f"\n[bold]All Plugins Metadata ({len(metadata_list)}):[/bold]")
        print(_pretty({"plugins_metadata": metadata_list}))

