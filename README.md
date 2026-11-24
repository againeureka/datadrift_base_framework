# datadrift_base_framework

**ddoc** (data doctor) — a plugin-based tool for detecting and
diagnosing data drift in ML systems, and leaving the evidence behind.

![ddoc diff demo](docs/imgs/ddoc_diff_demo.svg)

```bash
pip install -e packages/ddoc
ddoc diff baseline_data/ todays_data/     # OK / DRIFT / BLIND — exit code = verdict
```

## What is in this repository

```
packages/
├── ddoc/      # Core: the `ddoc` CLI, REST API and browser GUI (pluggy hookspecs + YAML recipes)
└── plugins/   # Plugins, each an independent pip package (vision, text, timeseries, ...)
examples/      # Runnable examples: Docker deployment, CI gate, application integration
run.sh         # One entry point: setup, demo, serve, test, gate, docker
```

Three ways to use it:

| Surface | Entry point | Use case |
|---|---|---|
| **CLI** | `ddoc <command>` | scripting, CI, remote/batch execution |
| **REST** | `ddoc serve` | integrate with your own services |
| **GUI** | `ddoc serve`, then open it in a browser | interactive exploration |

## Core ideas

- **The framework is light; plugins carry the weight.** `ddoc` core defines
  [pluggy hookspecs](packages/ddoc/ddoc/plugins/hookspecs.py)
  (`drift_detect`, `eda_run`, `report_render`, `export_drift_report`, ...).
  Adding a new detector means writing a plugin — zero framework changes.
- **An honest verdict.** `ddoc diff` answers `OK`, `DRIFT` or `BLIND` —
  "cannot judge" is a first-class answer rather than a silent OK.
- **Recipes.** A YAML DSL (`ddoc recipe run <file>`) chains fetch → analyze →
  report → export steps with variable substitution, so an entire drift-check
  pipeline is a reviewable text file.
- **Loose coupling to applications.** ddoc reads data through `ddoc fetch`
  and ships drift reports as a stable JSON envelope through `ddoc export`
  (file or HTTP) — no shared Python dependency in either direction.

## Quick start

```bash
./run.sh setup        # create ./.venv, install ddoc + categorical/tabular plugins
./run.sh doctor       # check the environment: plugins load? PDF, serve, embeddings available?
./run.sh menu         # pick what to do by situation, fill in values, run it
./run.sh demo         # toy data → verdict → HTML report in outputs/demo/
./run.sh serve        # REST API + GUI on http://127.0.0.1:8765
./run.sh help         # test, gate, docker, and running any ddoc command
```

`./run.sh setup all` installs every plugin (the vision plugin pulls in
torch). Reports and hints are in English by default; set `DDOC_LANG=ko`
(or pass `--lang ko`) for Korean, e.g. `DDOC_LANG=ko ./run.sh demo`. Without bash (e.g. Windows without WSL), the same steps by hand:

```bash
pip install -e packages/ddoc                              # core CLI
pip install -e packages/plugins/ddoc-plugin-categorical   # add a domain plugin

ddoc examples generate categorical --out /tmp/d --scenario shifted
ddoc diff /tmp/d/ref /tmp/d/cur                           # DRIFT, exit code 1
ddoc diff /tmp/d/ref /tmp/d/cur --report note.html
ddoc serve                                                # REST API + GUI on http://127.0.0.1:8765
```

**[Usage examples](packages/ddoc/docs/guides/usage.en.md)** ([한국어](packages/ddoc/docs/guides/usage.md))
walk through every common situation with real output: comparing two CSVs and reading the result,
checking before judging, profiling one dataset, images and text with the embedding map, sharing
reports, CI gates, and the GUI and REST API.

More in the [ddoc README](packages/ddoc/README.md), its [docs](packages/ddoc/docs/) and the
[cheat sheet](packages/ddoc/docs/cheatsheet.md) of commands by situation. What works, what is
partial and what is not verified yet is in the [status page](packages/ddoc/docs/status.md) (Korean).

## The diagnostic report

`ddoc diff ... --report note.html` writes one self-contained HTML file laid
out like an inspection report: what was compared, a summary with the
verdict, and a **pattern reading** — the measurements matched against
written-up drift patterns (regime shift, unit change, new categories, a
shift with a wider spread, ...), each with what it usually means, common
causes and what to check first. It opens offline, switches between day
and night, and prints to A4. Shown here for the
[Drift Zoo regime-shift case](packages/ddoc/zoo/regime_shift/):

![Diagnostic report: summary and pattern reading](docs/imgs/diagnostic-report.png)

A **data diagram** draws each column as one part with a baseline-vs-current
thumbnail and numbers the parts that need attention; each number leads
to that column's findings. Further down come the inspection table
(measured value, criterion and result per item), charts per attribute
and the items to check.

![Data diagram with numbered callouts](docs/imgs/data-diagram.png)

The same report comes out of `ddoc report render` (HTML, PDF, or Markdown
for pull requests) and the **Diagnose** tab of `ddoc serve`. For a single
dataset, `ddoc analyze eda` profiles a CSV and renders a data profile
report in the same format, with quality findings such as identifier-like
columns, far outliers or spelling variants:

![Data profile report](docs/imgs/profile-report.png)

Reports are in English by default; `--lang ko` (or `DDOC_LANG=ko`) writes
them in Korean.

## In-tree plugins

`vision` (image EDA / embedding drift) · `text` · `timeseries` · `audio` ·
`categorical` (JS-divergence on count dicts) · `tabular` · `evidently`
(third-party integration)

Each plugin is an independent pip package registered through the
`ddoc` entry-point group. The packages are not on PyPI yet, so install a
plugin from its folder (`pip install -e packages/plugins/ddoc-plugin-vision`)
or all of them with `./run.sh setup all`; `packages/plugins/ddoc-full` is the
meta-package for the common set once they are published.

## Use it as a CI data gate

`ddoc diff`'s exit code carries the verdict (0 = OK, 1 = DRIFT,
2 = BLIND), so it drops straight into CI — a linter for data:

```yaml
- uses: keti-datadrift/datadrift_base_framework@main
  with:
    ref: data/baseline
    cur: data/incoming
    report: drift_note.html   # self-contained evidence report
- uses: actions/upload-artifact@v4
  if: failure()
  with: { name: drift-note, path: drift_note.html }
```

The action installs the ddoc core, which judges CSV files and folders
on its own; see [examples/ci-gate](examples/ci-gate/) for other data
and for CI systems other than GitHub Actions.

Our own [CI](.github/workflows/ci.yml) dogfoods this on the
[Drift Zoo](packages/ddoc/zoo/) — a gallery of classic drift
patterns (regime shift, creeping degradation, and a rhythm-not-drift
trap case that must stay **OK**), each reproducible in one command and
pinned as a regression test.

## Examples

- [`examples/serve`](examples/serve/) — `ddoc serve` in Docker Compose
- [`examples/ci-gate`](examples/ci-gate/) — drift gate for GitHub Actions or any CI
- [`examples/app-integration`](examples/app-integration/) — `fetch` → `diff` → `export` to an application endpoint

## License

Apache License 2.0 — see [LICENSE](LICENSE).
Vendored third-party code retains its original license
(e.g. OpenAI CLIP under MIT, bundled in the vision and text plugins, see
`packages/plugins/ddoc-plugin-vision/ddoc_plugin_vision/clip/LICENSE`).
