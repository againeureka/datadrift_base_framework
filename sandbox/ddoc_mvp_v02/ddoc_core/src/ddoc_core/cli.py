from __future__ import annotations
import json
import typer
from .types import Manifest
from .utils import sha256_file, file_size
from .adapters.detect import detect_format
from .operators.eda import eda
from .registry import register_builtin_operators, list_operators as _list_ops
from .runtime_engine import run_operator

app = typer.Typer(add_completion=False, help="ddoc CLI (MVP) - ingest/eda for debugging & reproducibility")

@app.command()
def ingest(path: str, name: str = "dataset") -> None:
    """Create a simple manifest for a local file."""
    m = Manifest(
        dataset_name=name,
        files=[path],
        sha256=sha256_file(path),
        size_bytes=file_size(path),
        detected=detect_format(path),
    )
    typer.echo(m.model_dump_json(indent=2))

@app.command()
def eda_cmd(path: str) -> None:
    """Run simple EDA for a local file (CSV supported)."""
    r = eda(path)
    typer.echo(r.model_dump_json(indent=2))



@app.command("operators")
def list_operators_cmd() -> None:
    """List available operators (from registry)."""
    register_builtin_operators()
    ops = _list_ops()
    typer.echo(json.dumps([o.model_dump() for o in ops], ensure_ascii=False, indent=2))

@app.command("run")
def run_cmd(operator_name: str, paths: list[str] = typer.Argument(...), params_json: str = "{}") -> None:
    """Run an operator locally for debugging/repro."""
    params = json.loads(params_json)
    result = run_operator(operator_name, paths, params=params)
    typer.echo(result.model_dump_json(indent=2))


def main():
    app()

if __name__ == "__main__":
    main()
