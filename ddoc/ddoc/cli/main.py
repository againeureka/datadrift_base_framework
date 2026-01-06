# ddoc/cli/main.py
from __future__ import annotations
import typer
from typer.main import get_command
import logging
import os
import sys
import click
from pathlib import Path
from dotenv import load_dotenv
import contextlib

from importlib.metadata import version as get_version, metadata as get_metadata, PackageNotFoundError

# ddoc 패키지 내부 모듈은 가정하고 그대로 둡니다.
from ddoc.cli import commands as core_commands
from ddoc.cli.plugins import app as app_plugins
from ddoc.core.plugins import get_plugin_manager

# ------------------------------------------------------
# 📦 pyproject.toml 메타 정보 읽기
# ------------------------------------------------------
try:
    APP_VERSION = get_version("ddoc")
except PackageNotFoundError:
    APP_VERSION = "0.0.0"

try:
    meta = get_metadata("ddoc")
    DESCRIPTION = meta.get("Summary", "ddoc: data drift doctor")
except Exception:
    DESCRIPTION = "ddoc: data drift doctor"

RELEASE_DATE = ""         # 여전히 config에 있다면 별도 유지
DDOC_HUB_URL = ""         # 필요 시 상수 처리
ASCII_LOGO = r"""
=======================================
 _____    ____     ___     ____ 
|  __ \  |  _ \   / _ \   / ___| 
| |  | | | | | | | | | | |    
| |__| | | |_| | | |_| | | |___ 
|_____/  |____/   \___/   \____| 

Data Drift Doctor (ddoc)
Korea Electronics Technology Institute
=======================================
"""

# ------------------------------------------------------
# 🎨 로고 표시 함수
# ------------------------------------------------------
def show_logo():
    if ASCII_LOGO:
        click.echo(ASCII_LOGO)

# ------------------------------------------------------
# ⚙️ 공통 초기화 함수
# ------------------------------------------------------
def init_app(debug: bool = False, load_plugins: bool = True):
    load_dotenv()
    #click.echo("✅ .env 환경변수 로드됨")
    
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.debug("📋 로깅이 초기화되었습니다.")

    if debug:
        click.echo("🔬 디버그 모드 활성화: 상세 로그 (DEBUG 레벨)가 출력")
    
    # Only load plugins if needed (for performance)
    if load_plugins:
        get_plugin_manager()
        logging.debug("🔌 플러그인 매니저 로드됨.")

# ------------------------------------------------------
# 📘 메타 정보 출력
# ------------------------------------------------------
def print_meta_info(is_show_logo=True, full=False):
    if is_show_logo:
        show_logo()
        
    click.echo(f"🔖 Version       : {APP_VERSION}")
    if RELEASE_DATE:
        click.echo(f"📅 Release Date  : {RELEASE_DATE}")
    
    if full:
        click.echo(f"📘 Description   : {DESCRIPTION}")
        if DDOC_HUB_URL:
            click.echo(f"🌐 Hub URL       : {DDOC_HUB_URL}")
    raise typer.Exit()

# ------------------------------------------------------
# 🧭 Typer 앱 정의
# ------------------------------------------------------
app = typer.Typer(
    help=DESCRIPTION,
    add_completion=False,
)

@app.callback(invoke_without_command=True)
def _bootstrap(
    ctx: typer.Context,
    version: bool = typer.Option(
        None,
        "--version",
        help="Show version info and exit.",
        is_eager=True,
        callback=lambda v: print_meta_info(is_show_logo=False, full=False) if v else None,
    ),
    about: bool = typer.Option(
        None,
        "--about",
        help="Show full app meta info and exit.",
        is_eager=True,
        callback=lambda a: print_meta_info(is_show_logo=True, full=True) if a else None,
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
):
    # 🚀 OPTIMIZATION: Only load plugins for commands that actually need them
    # Commands that need plugins: analyze, exp (with run), plugin, vis
    # Commands that DON'T need plugins: init, add, snapshot, and ALL --help calls
    
    # Check if --help is in the command line args
    is_help_request = '--help' in sys.argv or '-h' in sys.argv
    
    # Determine if plugins are needed based on the subcommand
    # NOTE: 'plugin' and 'showcmd' removed - they don't need heavy plugin loading
    plugin_dependent_commands = {'analyze', 'exp', 'vis'}
    load_plugins = (
        ctx.invoked_subcommand in plugin_dependent_commands
        and not is_help_request  # Don't load plugins for help
    )
    
    init_app(debug=debug, load_plugins=load_plugins)

    if ctx.invoked_subcommand is None:
        show_logo()
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)

# ------------------------------------------------------
# 🔗 명령어 등록
# ------------------------------------------------------
core_commands.register(app)


# 🌳 트리 출력 명령
@app.command("showcmd")
def showcmd(level: int = typer.Option(3, "--level", "-l", help="Maximum tree level")):
    """
    Tree output of registered commands with rich formatting
    """
    from rich.console import Console
    from rich.tree import Tree
    from rich.text import Text
    from rich.panel import Panel
    from rich import box
    
    console = Console()
    click_command = get_command(app)

    def build_tree(command: click.Command, tree: Tree, current_level: int = 0):
        """Build rich tree structure from click commands"""
        if isinstance(command, click.Group):
            commands = list(command.commands.items())
            for i, (name, sub_cmd) in enumerate(commands):
                # Skip hidden/deprecated commands
                if getattr(sub_cmd, 'hidden', False) or getattr(sub_cmd, 'deprecated', False):
                    continue
                
                is_last = i == len(commands) - 1
                help_text = sub_cmd.help or ""
                
                # Create styled text for command name and help
                if help_text:
                    display_text = f"[bold cyan]{name}[/bold cyan] [dim]{help_text}[/dim]"
                else:
                    display_text = f"[bold cyan]{name}[/bold cyan]"
                
                # Special formatting for snapshot command (show more details)
                if name == "snapshot":
                    display_text = f"[bold cyan]{name}[/bold cyan] [dim]{help_text}[/dim]"
                    subtree = tree.add(display_text)
                    # Add snapshot subcommands as documentation
                    subtree.add("[green]create[/green] [dim]-m \"message\" -a alias[/dim]")
                    subtree.add("[green]list[/green] [dim]--list / --oneline[/dim]")
                    subtree.add("[green]show[/green] [dim]<version>[/dim]")
                    subtree.add("[green]restore[/green] [dim]--restore <version>[/dim]")
                    subtree.add("[green]compare[/green] [dim]--diff v1 v2[/dim]")
                    subtree.add("[green]graph[/green] [dim]--graph[/dim]")
                    subtree.add("[green]delete[/green] [dim]--delete <version>[/dim]")
                    subtree.add("[green]alias[/green] [dim]--set-alias / --unalias[/dim]")
                    subtree.add("[green]verify[/green] [dim]--verify / --verify-all[/dim]")
                    continue
                
                if current_level < level:
                    if isinstance(sub_cmd, click.Group):
                        # Create subtree for groups
                        subtree = tree.add(display_text)
                        build_tree(sub_cmd, subtree, current_level + 1)
                    else:
                        # Add leaf command
                        tree.add(display_text)
                else:
                    # Add collapsed representation for deep levels
                    tree.add(f"[dim]{name}...[/dim]")

    # Create main tree
    main_tree = Tree("🌳 [bold green]ddoc Command Structure[/bold green]", guide_style="dim")
    
    # Add root commands
    build_tree(click_command, main_tree, 0)
    
    # Display with panel
    console.print()
    console.print(Panel.fit(
        main_tree,
        title="[bold blue]ddoc Command Tree[/bold blue]",
        subtitle=f"[dim]Level limit: {level}[/dim]",
        border_style="blue",
        box=box.ROUNDED
    ))
    console.print()
    
    # Add usage examples (v2.0 style)
    console.print("[bold yellow]💡 Usage Examples (v2.0):[/bold yellow]")
    console.print()
    console.print("[bold]Getting Started:[/bold]")
    console.print("  [cyan]ddoc init myproject[/cyan]             # Initialize new workspace")
    console.print("  [cyan]ddoc add --data dataset.zip[/cyan]     # Add and extract data")
    console.print("  [cyan]ddoc add --code train.py[/cyan]        # Add training code")
    console.print()
    console.print("[bold]Snapshot Management:[/bold]")
    console.print("  [cyan]ddoc snapshot -m \"baseline\"[/cyan]     # Create snapshot")
    console.print("  [cyan]ddoc snapshot --list[/cyan]            # List all snapshots")
    console.print("  [cyan]ddoc snapshot v01[/cyan]               # Show snapshot details")
    console.print("  [cyan]ddoc snapshot --restore v01[/cyan]     # Restore snapshot")
    console.print("  [cyan]ddoc snapshot --diff v01 v02[/cyan]    # Compare snapshots")
    console.print("  [cyan]ddoc snapshot --graph[/cyan]           # Show lineage graph")
    console.print()
    console.print("[bold]Analysis & Experiments:[/bold]")
    console.print("  [cyan]ddoc analyze eda my_data[/cyan]        # Run EDA analysis")
    console.print("  [cyan]ddoc analyze drift d1 d2[/cyan]        # Drift detection")
    console.print("  [cyan]ddoc exp run my_data[/cyan]            # Run experiment")
    console.print()
    console.print("[bold]System:[/bold]")
    console.print("  [cyan]ddoc plugin list[/cyan]                # List installed plugins")
    console.print("  [cyan]ddoc vis[/cyan]                        # Launch GUI")
    console.print()

# ------------------------------------------------------
# 🚀 엔트리포인트
# ------------------------------------------------------
def main():
    try:
        
        app()
    except typer.Exit:
        raise
    except Exception as e:
        logging.exception("❌ 처리되지 않은 예외 발생:")
        click.echo(f"❌ 에러: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    main()