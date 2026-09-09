"""
map_structure.py
=================
Prints the real folder structure under a given root -- file counts per
folder rather than every filename (some of these folders have thousands of
tiles), plus explicit call-outs for the specific files that matter for
path resolution (config.py, tile_metadata.csv, classes.txt, .here markers,
label folders). No project dependencies -- stdlib only, runs in any venv.

Usage:
    python map_structure.py "C:\\Users\\AMURRA02\\...\\Github\\tp_qa"
    python map_structure.py "C:\\Users\\AMURRA02\\...\\Github\\tp_qa" --max-depth 4
"""
import argparse
import sys
from pathlib import Path

KEY_FILES = {"config.py", "tile_metadata.csv", "classes.txt", ".here",
             "label_app.R", "dataset.yaml"}
KEY_DIRS = {"labels", "labels_", "rgb", "ndwi", "png", "ls_export"}


def summarize_dir(path: Path) -> str:
    try:
        entries = list(path.iterdir())
    except PermissionError:
        return "  [permission denied]"
    files = [e for e in entries if e.is_file()]
    dirs = [e for e in entries if e.is_dir()]
    key_files_here = [f.name for f in files if f.name in KEY_FILES]
    ext_counts = {}
    for f in files:
        ext_counts[f.suffix or "(no ext)"] = ext_counts.get(f.suffix or "(no ext)", 0) + 1
    ext_summary = ", ".join(f"{n} {ext}" for ext, n in sorted(ext_counts.items(), key=lambda x: -x[1])[:4])
    parts = []
    if ext_summary:
        parts.append(ext_summary)
    if key_files_here:
        parts.append(f"KEY FILES: {', '.join(key_files_here)}")
    return "  (" + "; ".join(parts) + ")" if parts else ""


def walk(path: Path, prefix: str = "", depth: int = 0, max_depth: int = 5):
    if depth > max_depth:
        return
    try:
        entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    except PermissionError:
        print(f"{prefix}[permission denied]")
        return

    dirs = [e for e in entries if e.is_dir() and not e.name.startswith(".git")]
    # hidden marker files (like .here) are important -- show them explicitly
    hidden_markers = [e for e in entries if e.is_file() and e.name.startswith(".")]

    for m in hidden_markers:
        print(f"{prefix}{m.name}  <-- marker file")

    for d in dirs:
        n_files = sum(1 for e in d.iterdir() if e.is_file()) if d.exists() else 0
        summary = summarize_dir(d)
        print(f"{prefix}{d.name}/{summary}")
        walk(d, prefix + "    ", depth + 1, max_depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="folder to map")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--out", type=str, default="structure_map.txt",
                     help="output text file (default: structure_map.txt, written "
                          "next to this script)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        sys.exit(1)

    out_path = Path(args.out)
    lines = []

    def emit(s=""):
        lines.append(s)

    emit(f"=== {root} ===")
    emit()

    # Redirect walk()'s prints into `lines` instead of stdout, without
    # rewriting walk() itself -- simplest way to keep one code path for both
    # the (removed) console version and the file version.
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        walk(root, max_depth=args.max_depth)
    lines.extend(buf.getvalue().splitlines())

    emit()
    emit("=== Key files found anywhere under root ===")
    for kf in sorted(KEY_FILES):
        matches = list(root.rglob(kf))
        for m in matches:
            emit(f"  {m.relative_to(root)}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {out_path.resolve()}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()