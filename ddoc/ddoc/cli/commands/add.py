"""File addition command"""
from typing import Optional
import typer
from rich import print


def add(
    data: Optional[str] = typer.Option(None, "--data", help="Add data file or directory"),
    code: Optional[str] = typer.Option(None, "--code", help="Add code file"),
    notebook: Optional[str] = typer.Option(None, "--notebook", help="Add notebook file"),
    trainer: Optional[str] = typer.Option(
        None,
        "--trainer",
        help="Trainer type/name for this code (required with --code). e.g. yolo, custom",
    ),
):
    """
    Add files to the ddoc workspace (data/code/notebook).
    
    Automatically handles file copying, DVC tracking for data, and Git tracking for code.
    Supports zip/tar.gz extraction for data files.
    
    Examples:
        ddoc add --data datasets/train.zip
        ddoc add --data datasets/raw_images/
        ddoc add --code scripts/train.py
        ddoc add --notebook analysis.ipynb
    """
    from ddoc.core.file_service import get_file_service
    
    if not any([data, code, notebook]):
        print("[red]❌ Please specify at least one of: --data, --code, or --notebook[/red]")
        raise typer.Exit(code=1)
    
    # Validate trainer option usage
    if trainer and not code:
        print("[red]❌ --trainer option can only be used together with --code[/red]")
        raise typer.Exit(code=1)
    if code and not trainer:
        print("[red]❌ --trainer is required when adding code.[/red]")
        print("   e.g. --trainer yolo   # built-in YOLO example")
        print("        --trainer custom # user-defined trainer")
        raise typer.Exit(code=1)
    
    file_service = get_file_service()
    
    # Add data
    if data:
        print(f"[cyan]📦 Adding data from: {data}[/cyan]")
        result = file_service.add_data(data)
        
        if result["success"]:
            print(f"[green]✅ Data added successfully[/green]")
            for item in result["added_items"]:
                print(f"   📁 {item}")
            
            if result.get("dvc_tracked"):
                print(f"[green]✅ DVC tracking enabled[/green]")
            if result.get("git_staged"):
                print(f"[green]✅ data.dvc staged in git[/green]")
        else:
            print(f"[red]❌ Failed to add data: {result['error']}[/red]")
            raise typer.Exit(code=1)
    
    # Add code
    if code:
        print(f"[cyan]💻 Adding code from: {code}[/cyan]")
        if trainer:
            print(f"[cyan]🧩 Trainer: {trainer} (code/trainers/{trainer}/)[/cyan]")
        result = file_service.add_code(code, trainer=trainer)
        
        if result["success"]:
            print(f"[green]✅ Code added: {result['added_file']}[/green]")
            if result.get("git_staged"):
                print(f"[green]✅ Code staged in git[/green]")
        else:
            print(f"[red]❌ Failed to add code: {result['error']}[/red]")
            raise typer.Exit(code=1)
    
    # Add notebook
    if notebook:
        print(f"[cyan]📓 Adding notebook from: {notebook}[/cyan]")
        result = file_service.add_notebook(notebook)
        
        if result["success"]:
            print(f"[green]✅ Notebook added: {result['added_file']}[/green]")
            if result.get("git_staged"):
                print(f"[green]✅ Notebook staged in git[/green]")
        else:
            print(f"[red]❌ Failed to add notebook: {result['error']}[/red]")
            raise typer.Exit(code=1)
    
    print()

