#!/usr/bin/env python3
"""
setup_positron_workspace.py

Configures the tp_qa workspace root so Positron correctly resolves the
review_app venv/interpreter and imports when opened at the tp_qa level.

Run this from the tp_qa repo root:
    python setup_positron_workspace.py
Safe to re-run.
"""
import json
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE_ROOT = Path.cwd()
VSCODE_DIR = WORKSPACE_ROOT / ".vscode"
SETTINGS_FILE = VSCODE_DIR / "settings.json"

# venv layout differs on Windows vs Mac/Linux
venv_python_rel = "review_app/.venv/Scripts/python.exe" if platform.system() == "Windows" \
    else "review_app/.venv/bin/python"
venv_python_abs = WORKSPACE_ROOT / venv_python_rel

print(f"Workspace root: {WORKSPACE_ROOT}")

if not venv_python_abs.exists():
    sys.exit(
        f"ERROR: expected venv python not found at {venv_python_abs}\n"
        f"Run this script from the tp_qa repo root, and confirm review_app/.venv exists."
    )

VSCODE_DIR.mkdir(exist_ok=True)

if SETTINGS_FILE.exists():
    backup = SETTINGS_FILE.with_name(f"settings.json.bak-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy(SETTINGS_FILE, backup)
    print(f"Backed up existing settings.json -> {backup.name}")

    content = SETTINGS_FILE.read_text().strip()
    try:
        settings = json.loads(content) if content else {}
    except json.JSONDecodeError:
        print(f"\nWARNING: {SETTINGS_FILE} has invalid JSON -- not touching it.")
        print("Merge these keys in manually:")
        print(json.dumps({
            "python.defaultInterpreterPath": f"${{workspaceFolder}}/{venv_python_rel}",
            "python.analysis.extraPaths": ["review_app"]
        }, indent=4))
        sys.exit(1)
else:
    settings = {}

settings["python.defaultInterpreterPath"] = f"${{workspaceFolder}}/{venv_python_rel}"

extra_paths = settings.get("python.analysis.extraPaths", [])
if "review_app" not in extra_paths:
    extra_paths.append("review_app")
settings["python.analysis.extraPaths"] = extra_paths

SETTINGS_FILE.write_text(json.dumps(settings, indent=4) + "\n")

print(f"\nWrote {SETTINGS_FILE}")
print("\n=== Done ===")
print("Next steps:")
print("  1. Reload Positron (Cmd/Ctrl+Shift+P -> 'Reload Window') to pick up the new interpreter.")
print("  2. For module-style commands (python -m backend.queue_loader ...), 'cd review_app'")
print("     first -- this script fixes the EDITOR's interpreter/import resolution,")
print("     not your terminal's working directory.")