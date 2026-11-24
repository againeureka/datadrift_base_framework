"""How to install a plugin, in a form that works today.

The plugins are not published to PyPI, so installing one by its name
finds nothing. When ddoc runs from a checkout of this repository (an
editable install), the plugin's own folder sits next to it; otherwise pip
can install it straight from the repository with git.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_URL = "https://github.com/keti-datadrift/datadrift_base_framework"


def plugin_folder(kind: str) -> Path:
    """Where ``ddoc-plugin-<kind>`` would sit in a checkout of the repository."""
    return Path(__file__).resolve().parent.parent.parent / "plugins" / f"ddoc-plugin-{kind}"


def plugin_command(kind: str) -> str:
    """The pip command that installs the plugin for ``kind`` (vision, text, ...)."""
    folder = plugin_folder(kind)
    if (folder / "pyproject.toml").is_file():
        try:  # short and copy-pasteable from inside the checkout
            shown = "." + os.sep + str(folder.relative_to(Path.cwd().resolve()))
        except ValueError:
            shown = str(folder)
        return f"pip install -e {shown}"
    return f'pip install "git+{REPO_URL}#subdirectory=packages/plugins/ddoc-plugin-{kind}"'
