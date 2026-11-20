#!/usr/bin/env bash
# One entry point for working with this repository.
#
#   ./run.sh help
#
# Everything runs inside a virtual environment at ./.venv (override with
# DDOC_VENV). Create it with `./run.sh setup`; later commands use it
# automatically, so there is nothing to activate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${DDOC_VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3}"
DEFAULT_PLUGINS=(categorical tabular)
ALL_PLUGINS=(categorical tabular timeseries text audio evidently vision)

# Tests the CI runs on every push; `./run.sh test --all` runs everything.
CI_TESTS=(
  tests/test_diff_cli.py
  tests/test_zoo.py
  tests/test_plugin_categorical.py
  tests/test_examples_generate.py
  tests/test_recipe_catalog.py
  tests/test_docs.py
)

say() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage: ./run.sh <command> [args]

  setup [plugin ...]     Create ./.venv and install ddoc plus plugins
                         (default: categorical tabular; "all" for every plugin:
                          categorical tabular timeseries text audio evidently vision)
  doctor                 Check the environment: venv, Python, plugins (do they
                         load?), optional features, Docker and Chrome
  menu                   Pick what to do by situation, fill in values, run it
                         (the same catalog as `ddoc scenarios` and the GUI)
  demo                   Generate toy data, judge it, and write an HTML report
  serve [ddoc serve args]
                         Start the REST API + browser GUI (http://127.0.0.1:8765)
  test [--all]           Run the CI test set (or the whole suite)
  gate <ref> <cur> [report.html]
                         Drift gate: exit 0 OK, 1 DRIFT, 2 BLIND
  docker [up|down]       Build and run `ddoc serve` with Docker Compose
  docs [build|check|shots]
                         Regenerate the doc parts that come from code (build),
                         check they are current (check, also run by test), or
                         re-take the README screenshots (shots: needs Chrome)
  ddoc [args]            Run any ddoc command inside the environment
  help                   Show this message

Environment: DDOC_VENV (venv path), PYTHON (interpreter used by setup).
EOF
}

require_venv() {
  [ -x "$VENV/bin/ddoc" ] || die "no ddoc environment at $VENV — run ./run.sh setup first"
  export PATH="$VENV/bin:$PATH"
}

plugin_installed() {
  "$VENV/bin/python" -c "import ddoc_plugin_$1" 2>/dev/null
}

cmd_setup() {
  local plugins=("$@")
  [ ${#plugins[@]} -eq 0 ] && plugins=("${DEFAULT_PLUGINS[@]}")
  [ "${plugins[0]}" = "all" ] && plugins=("${ALL_PLUGINS[@]}")

  local args=(-e "$ROOT/packages/ddoc")
  local p
  for p in "${plugins[@]}"; do
    [ -d "$ROOT/packages/plugins/ddoc-plugin-$p" ] \
      || die "unknown plugin '$p' (available: ${ALL_PLUGINS[*]})"
    args+=(-e "$ROOT/packages/plugins/ddoc-plugin-$p")
  done

  "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 9))' \
    || die "Python 3.9 or newer is required ($PYTHON is $("$PYTHON" -V 2>&1))"

  if [ ! -x "$VENV/bin/python" ]; then
    say "Creating virtual environment at $VENV"
    "$PYTHON" -m venv "$VENV"
  fi
  say "Installing ddoc and plugins: ${plugins[*]}"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  "$VENV/bin/python" -m pip install -q "${args[@]}"
  "$VENV/bin/ddoc" --version
  say "Done. Try: ./run.sh demo"
}

cmd_demo() {
  require_venv
  plugin_installed categorical \
    || die "the demo uses the categorical plugin — run ./run.sh setup (or ./run.sh setup categorical)"

  local out="$ROOT/outputs/demo"
  rm -rf "$out"
  mkdir -p "$out"
  # Plugins write their working files under ./analysis, so run in $out
  # to keep everything the demo produces in one place.
  cd "$out"

  say "1/3  Generating a shifted toy dataset pair in $out"
  ddoc examples generate categorical --out . --scenario shifted > /dev/null

  say "2/3  Judging it: ddoc diff ref cur"
  local code=0
  ddoc diff ref cur || code=$?

  say "3/3  Writing the evidence report"
  ddoc diff ref cur --report note.html > /dev/null || true

  echo
  echo "verdict exit code : $code  (0 OK, 1 DRIFT, 2 BLIND)"
  echo "report            : $out/note.html"
  echo "data              : $out/ref, $out/cur"
}

cmd_serve() {
  require_venv
  exec ddoc serve "$@"
}

cmd_test() {
  require_venv
  if ! "$VENV/bin/python" -c "import pytest, httpx" 2>/dev/null; then
    say "Installing test dependencies (pytest, httpx2)"
    "$VENV/bin/python" -m pip install -q pytest httpx2
  fi
  cd "$ROOT/packages/ddoc"
  if [ "${1:-}" = "--all" ]; then
    python -m pytest -q
  else
    python -m pytest -q "${CI_TESTS[@]}"
  fi
}

cmd_gate() {
  require_venv
  [ $# -ge 2 ] || die "usage: ./run.sh gate <ref> <cur> [report.html]"
  exec "$ROOT/examples/ci-gate/gate.sh" "$@"
}

cmd_docker() {
  local compose
  if docker compose version > /dev/null 2>&1; then
    compose=(docker compose)
  elif command -v docker-compose > /dev/null 2>&1; then
    compose=(docker-compose)
  else
    die "Docker Compose is not available"
  fi
  docker info > /dev/null 2>&1 || die "the Docker daemon is not running"

  cd "$ROOT/examples/serve"
  case "${1:-up}" in
    up)
      mkdir -p data
      if [ ! -f .env ]; then
        echo "DDOC_API_KEY=$("${PYTHON}" -c 'import secrets; print(secrets.token_hex(16))')" > .env
        say "Wrote an API key to examples/serve/.env"
      fi
      "${compose[@]}" up --build -d
      say "ddoc serve is starting on http://127.0.0.1:8765 (API key in examples/serve/.env)"
      ;;
    down)
      "${compose[@]}" down
      ;;
    *) die "usage: ./run.sh docker [up|down]" ;;
  esac
}

cmd_doctor() {
  if [ ! -x "$VENV/bin/ddoc" ]; then
    printf '  ✕ no ddoc environment at %s\n      ⤷ ./run.sh setup\n' "$VENV"
    return 1
  fi
  printf 'Environment: %s\n' "$VENV"
  local rc=0
  "$VENV/bin/ddoc" doctor || rc=$?
  echo
  printf 'Tools used by run.sh\n'
  if command -v docker >/dev/null 2>&1; then
    printf '  ✓ docker  (./run.sh docker)\n'
  else
    printf '  ! docker not found — only ./run.sh docker needs it\n'
  fi
  if [ -n "${DDOC_CHROME:-}" ] || [ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ] \
      || command -v google-chrome >/dev/null 2>&1 || command -v chromium >/dev/null 2>&1; then
    printf '  ✓ chrome  (./run.sh docs shots)\n'
  else
    printf '  ! chrome not found — only ./run.sh docs shots needs it (or set DDOC_CHROME)\n'
  fi
  return "$rc"
}

cmd_docs() {
  require_venv
  local what="${1:-check}"
  if [ "$what" = "shots" ] && ! "$VENV/bin/python" -c "import PIL" 2>/dev/null; then
    say "Installing Pillow for the screenshots"
    "$VENV/bin/python" -m pip install -q Pillow
  fi
  "$VENV/bin/python" "$ROOT/packages/ddoc/scripts/docs.py" "$what"
}

cmd_menu() {
  require_venv
  exec ddoc scenarios menu "$@"
}

cmd_ddoc() {
  require_venv
  exec ddoc "$@"
}

main() {
  local cmd="${1:-help}"
  [ $# -gt 0 ] && shift
  case "$cmd" in
    setup)  cmd_setup "$@" ;;
    demo)   cmd_demo "$@" ;;
    serve)  cmd_serve "$@" ;;
    test)   cmd_test "$@" ;;
    gate)   cmd_gate "$@" ;;
    docker) cmd_docker "$@" ;;
    docs)   cmd_docs "$@" ;;
    doctor) cmd_doctor "$@" ;;
    menu)   cmd_menu "$@" ;;
    ddoc)   cmd_ddoc "$@" ;;
    help|-h|--help) usage ;;
    *) usage >&2; exit 2 ;;
  esac
}

main "$@"
