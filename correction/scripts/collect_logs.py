"""
collect_logs.py
===============
Copies SLURM job logs out of correction/logs/ (gitignored) into
correction/diagnostics/logs/ (not gitignored), so a log can be committed and
read on a machine that cannot see the cluster.

    python collect_logs.py --training              # every model-training log
    python collect_logs.py --pattern "04_*"        # newest 3 Stage 2 logs
    python collect_logs.py --latest 8              # newest 8 of anything
    python collect_logs.py --job 48213             # every file for one job

    --training is the one to reach for round over round: it takes the newest
    log for EACH of 03, 04, 06, 06b, 07, 07b and 12, so one call captures the
    whole modelling picture for a round. Comparing those across rounds is what
    exposed the spatial-CV fold imbalance -- no single run made it obvious.

WHY
    collect_diagnostics.py answers "what does the pipeline look like right
    now" and includes only a 45-line tail of recent logs, which is enough to
    see that something failed and never enough to see why. When the question
    is about the CONTENT of a run -- fold sizes, class counts, the metrics
    table, a traceback in context -- the whole log has to travel.

    Same channel, same reason: the HPC and the machine doing the analysis are
    different computers with no shared filesystem, and git already works in
    both directions.

WHAT IT DOES NOT DO
    It does not move or delete anything. correction/logs/ is left exactly as
    it was; this only ever reads and copies.

SIZE
    Logs from a hyperparameter search are mostly repetition, and a few are
    tens of MB. Two things keep the copies small enough to commit without
    thinking about it:

      * progress-bar lines (anything written with \\r -- tqdm, ultralytics,
        joblib) are collapsed to their final state, which is the only part
        that carries information once the bar has finished.
      * a file over --max-kb is truncated from the MIDDLE, keeping the head
        and the tail. That is deliberate: the head holds the inputs, row
        counts and fold construction, the tail holds the results and any
        traceback. The middle of a 20 MB search log is the part nobody reads.

    A truncation is always announced in the copy itself, so a reader can
    never mistake a shortened log for a complete one.
"""
import argparse
import datetime as dt
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

OUT_DIR = C.ROOT / "diagnostics" / "logs"
MAX_LINE = 400          # characters; long lines are almost always progress bars
HEAD_FRACTION = 0.6     # of the budget kept from the top, rest from the bottom


def clean_lines(raw: str, strip_progress: bool):
    """Collapse carriage-return progress output and clip absurd lines.

    A tqdm/ultralytics bar writes many updates to ONE line separated by \\r.
    Keeping the last segment keeps the finished bar and discards every
    intermediate frame, which is where the bulk of a big log actually is.
    """
    out = []
    # split("\n"), NOT splitlines(): splitlines() treats a bare \r as a line
    # break of its own, which shatters a progress bar into one line per frame
    # and leaves nothing for the collapse below to find. That is the whole
    # problem this function exists to solve, so it has to see the \r itself.
    for line in raw.split("\n"):
        line = line.rstrip("\r")        # normalize CRLF before looking inside
        if strip_progress and "\r" in line:
            line = line.split("\r")[-1]
        if len(line) > MAX_LINE:
            line = line[:MAX_LINE] + f"  ...[{len(line) - MAX_LINE} chars clipped]"
        out.append(line)
    return out


def truncate(lines, max_kb: int):
    """Keep the head and the tail, drop the middle, say so in place."""
    budget = max_kb * 1024
    size = sum(len(l) + 1 for l in lines)
    if size <= budget:
        return lines, False
    n_head = int(len(lines) * HEAD_FRACTION)
    head, tail, kept = [], [], 0
    for l in lines[:n_head]:
        if kept + len(l) > budget * HEAD_FRACTION:
            break
        head.append(l)
        kept += len(l) + 1
    for l in reversed(lines):
        if kept + len(l) > budget:
            break
        tail.append(l)
        kept += len(l) + 1
    tail.reverse()
    dropped = len(lines) - len(head) - len(tail)
    marker = [
        "",
        "=" * 78,
        f"[collect_logs.py TRUNCATED {dropped} line(s) here -- the file was "
        f"{size / 1024:.0f} KB, over the {max_kb} KB limit.",
        " Head and tail are kept because the inputs are at the top and the",
        " results are at the bottom. Re-copy with a larger --max-kb if the",
        " middle matters, or read the original on the cluster:",
        f" {C.LOGS_DIR}]",
        "=" * 78,
        "",
    ]
    return head + marker + tail, True


# The logs that describe a MODEL, as opposed to a data-prep step: class
# counts, fold construction, CV scores, the metrics table, feature importance,
# thresholds, and the holdout score. These are the ones worth keeping round
# over round -- reviewing them across three runs is what exposed the spatial-CV
# fold imbalance, which no single run made obvious.
TRAINING_PATTERNS = [
    "03_*.log",          # Stage 1
    "04_*.log",          # Stage 2a
    "07_stage2b_*.log",  # Stage 2b
    "07b_rerank_*.log",  # re-ranker
    "06_s2b_*.log",      # Stage 2b training-set build
    "06b_rerank_*.log",  # re-ranker training-set build
    "12_holdout_*.log",  # the honest read
]


def select(args) -> list[Path]:
    if not C.LOGS_DIR.exists():
        raise SystemExit(f"No log directory at {C.LOGS_DIR}")
    if args.training:
        # Newest --latest per PATTERN, not overall: one run of 04 must not
        # crowd out 12's only log. This is the round-over-round snapshot.
        picked = []
        for pat in TRAINING_PATTERNS:
            hits = sorted(C.LOGS_DIR.glob(pat),
                          key=lambda p: p.stat().st_mtime, reverse=True)
            picked.extend(hits[:args.latest])
        if not picked:
            raise SystemExit(
                f"No training logs in {C.LOGS_DIR}. Expected one of: "
                f"{', '.join(TRAINING_PATTERNS)}")
        return sorted(set(picked), key=lambda p: p.name)
    if args.job:
        hits = sorted(C.LOGS_DIR.glob(f"*{args.job}*"))
        if not hits:
            raise SystemExit(f"No log file mentions job {args.job} in its name.")
        return hits
    pool = sorted(C.LOGS_DIR.glob(args.pattern),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not pool:
        raise SystemExit(f"Nothing matches {args.pattern!r} in {C.LOGS_DIR}")
    return pool[:args.latest]


def main():
    ap = argparse.ArgumentParser(
        description="Copy job logs into correction/diagnostics/logs/ so they "
                    "can be committed and read elsewhere.")
    ap.add_argument("--pattern", default="*.log",
                    help='glob against correction/logs/, e.g. "04_*" or '
                         '"02_*_7.log" (default: *.log)')
    ap.add_argument("--latest", type=int, default=3,
                    help="how many of the newest matches to copy (default 3)")
    ap.add_argument("--job", default=None,
                    help="copy every log whose filename contains this job id, "
                         "ignoring --pattern and --latest")
    ap.add_argument("--training", action="store_true",
                    help="copy the newest --latest log for EACH model-training "
                         "step (03, 04, 06, 06b, 07, 07b, 12) instead of using "
                         "--pattern. The round-over-round snapshot: comparing "
                         "these across runs is what surfaced the spatial-CV "
                         "fold imbalance.")
    ap.add_argument("--max-kb", type=int, default=512,
                    help="per-file cap; over this the middle is dropped "
                         "(default 512)")
    ap.add_argument("--keep-progress", action="store_true",
                    help="do NOT collapse \\r progress-bar output. Only useful "
                         "if the bar itself is what you are debugging.")
    ap.add_argument("--clear", action="store_true",
                    help="delete previously copied logs first, so the "
                         "directory holds only this batch")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = select(args)

    print("=== collect_logs.py ===")
    print(f"from : {C.LOGS_DIR}")
    print(f"to   : {OUT_DIR}")
    print(f"\n{len(files)} file(s) selected:\n")

    if args.dry_run:
        for f in files:
            ts = dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="minutes")
            print(f"  {f.name:<34} {f.stat().st_size / 1024:>9.0f} KB  {ts}")
        print("\n--dry-run: nothing copied.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for old in OUT_DIR.glob("*.log"):
            old.unlink()
        for old in OUT_DIR.glob("MANIFEST.txt"):
            old.unlink()
        print("  --clear: previous copies removed\n")

    if args.training:
        how = f"--training presets (latest {args.latest} each)"
    elif args.job:
        how = f"job {args.job}"
    else:
        how = f"{args.pattern} (latest {args.latest})"
    manifest = [
        "collect_logs.py",
        f"host    : {platform.node()}",
        f"when    : {dt.datetime.now().isoformat(timespec='seconds')}",
        f"source  : {C.LOGS_DIR}",
        f"selected: {how}",
        "",
    ]

    total = 0
    for f in files:
        try:
            # read_bytes, NOT read_text: text mode uses universal newlines,
            # which converts a bare \r to \n on the way in and destroys the
            # progress-bar structure before clean_lines can collapse it.
            raw = f.read_bytes().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  SKIP {f.name}: unreadable ({e})")
            manifest.append(f"  SKIPPED {f.name}: {e}")
            continue
        lines = clean_lines(raw, strip_progress=not args.keep_progress)
        lines, was_cut = truncate(lines, args.max_kb)
        dest = OUT_DIR / f.name
        # newline="" so LF stay LF on Windows too -- these files get committed,
        # and a CRLF copy of a cluster log is noise in every diff.
        with dest.open("w", encoding="utf-8", newline="") as fh:
            fh.write("\n".join(lines) + "\n")
        kb_in, kb_out = f.stat().st_size / 1024, dest.stat().st_size / 1024
        total += kb_out
        src_ts = dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="minutes")
        note = "  TRUNCATED" if was_cut else ""
        print(f"  {f.name:<34} {kb_in:>8.0f} KB -> {kb_out:>7.0f} KB{note}")
        manifest.append(f"  {f.name:<34} {kb_in:>8.0f} KB -> {kb_out:>7.0f} KB  "
                        f"(job wrote it {src_ts}){note}")

    manifest += ["", f"total copied: {total:.0f} KB",
                 "", "Originals are untouched. This directory is not gitignored."]
    (OUT_DIR / "MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    print(f"\ntotal: {total:.0f} KB in {OUT_DIR}")
    print("\nCommit and push them:")
    print(f"  git add {OUT_DIR.relative_to(C.ROOT.parent)}")
    print('  git commit -m "logs for review"')
    print("  git push")


if __name__ == "__main__":
    main()
