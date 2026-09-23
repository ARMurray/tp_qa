"""
collect_diagnostics.py
=======================
Writes one text file describing the state of the pipeline on this machine, so
it can be committed and read somewhere else.

    python collect_diagnostics.py
    -> correction/diagnostics/diag_<host>_<YYYYMMDD-HHMMSS>.txt

WHY
    The HPC and the machine doing the analysis are different computers with no
    shared filesystem, and the login console is not always a place where things
    can be run interactively. Git is the channel that already works in both
    directions, so: run this as a job, commit the file it writes, push. The
    other end pulls and reads.

    correction/diagnostics/ is deliberately NOT gitignored, unlike logs/ and
    data/. Everything written here is small, textual, and meant to travel.

WHAT IT COLLECTS
    Environment, config paths and whether they exist, parcel-store coverage
    against the training universe, the deployed detector's classes and age,
    detection output inventory per state, feature-table presence, and the tail
    of the most recent job logs.

    All of it read-only. Nothing here writes to the pipeline's data, and a
    failure in any one section is caught and reported rather than killing the
    rest -- a partial diagnostic is still worth reading, and the section that
    failed is itself a finding.

PRIVACY / SIZE
    Log tails are capped and only the most recent few files are read, so the
    output stays a few hundred KB at worst. It contains paths, counts and
    error text -- no imagery, no parcel data, no coordinates beyond what a
    summary line prints.
"""
import datetime as dt
import io
import os
import platform
import subprocess
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

OUT_DIR = C.ROOT / "diagnostics"
LOG_TAIL_LINES = 45
LOG_FILES_PER_PATTERN = 3


class Report:
    def __init__(self):
        self.buf = io.StringIO()

    def h(self, title):
        self.buf.write(f"\n{'=' * 78}\n{title}\n{'=' * 78}\n")

    def p(self, *a):
        self.buf.write(" ".join(str(x) for x in a) + "\n")

    def section(self, title, fn):
        """Run one check. A failure is reported in place and does not stop the
        rest -- the whole point is to come back with something readable."""
        self.h(title)
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                fn(self)
            self.buf.write(out.getvalue())
        except Exception:
            self.p("*** this section failed ***")
            self.p(traceback.format_exc(limit=6))


def sec_environment(r):
    r.p(f"host          : {platform.node()}")
    r.p(f"when          : {dt.datetime.now().isoformat(timespec='seconds')}")
    r.p(f"python        : {sys.version.split()[0]}  ({sys.executable})")
    r.p(f"cwd           : {os.getcwd()}")
    r.p(f"ROOT          : {C.ROOT}")
    for var in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK"):
        if os.environ.get(var):
            r.p(f"{var:<14}: {os.environ[var]}")
    for cmd, label in ((["git", "rev-parse", "--short", "HEAD"], "git HEAD"),
                       (["git", "rev-parse", "--abbrev-ref", "HEAD"], "git branch"),
                       (["git", "status", "--porcelain"], "git dirty")):
        try:
            out = subprocess.run(cmd, cwd=str(C.ROOT.parent), capture_output=True,
                                 text=True, timeout=15)
            val = out.stdout.strip() or "(clean)"
            r.p(f"{label:<14}: {val.splitlines()[0] if val.splitlines() else val}"
                + (f"  (+{len(val.splitlines()) - 1} more)" if len(val.splitlines()) > 1 else ""))
        except Exception as e:
            r.p(f"{label:<14}: unavailable ({type(e).__name__})")


def sec_paths(r):
    checks = [
        ("PARCEL_BASE", C.PARCEL_BASE), ("NLCD_PATH", C.NLCD_PATH),
        ("CENSUS_GDB", C.CENSUS_GDB), ("CWNS_DIR", C.CWNS_DIR),
        ("TRAINING_GPKG", C.TRAINING_GPKG), ("OSM_PATH", C.OSM_PATH),
        ("OD_MODEL_DIR", C.OD_MODEL_DIR), ("NLCD_OUTPUT_DIR", C.NLCD_OUTPUT_DIR),
        ("OD_OUTPUT_DIR", C.OD_OUTPUT_DIR),
        ("OD_OUTPUT_DIR_CORRECTED", C.OD_OUTPUT_DIR_CORRECTED),
        ("FEATURES_OUTPUT_DIR", C.FEATURES_OUTPUT_DIR),
        ("FEATURE_SHARD_DIR", getattr(C, "FEATURE_SHARD_DIR", None)),
    ]
    for name, p in checks:
        if p is None:
            r.p(f"  {name:<26} (not defined in config)")
            continue
        mark = "OK     " if Path(p).exists() else "MISSING"
        r.p(f"  {mark} {name:<26} {p}")


def sec_model(r):
    pts = sorted(C.OD_MODEL_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime,
                 reverse=True) if C.OD_MODEL_DIR.exists() else []
    if not pts:
        r.p("  no .pt under OD_MODEL_DIR -- 01b/01c/01e cannot run")
        return
    m = pts[0]
    ts = dt.datetime.fromtimestamp(m.stat().st_mtime)
    r.p(f"  deployed   : {m}")
    r.p(f"  mtime      : {ts.isoformat(timespec='seconds')}  ({m.stat().st_size/1e6:.1f} MB)")
    blob = m.read_bytes()
    names = ["aeration_basin", "chlorine_contact", "clarifier",
             "digester", "drying_bed", "oxidation_pond"]
    present = [n for n in names if n.encode() in blob]
    r.p(f"  classes    : {len(present)}/6 -> {present}")
    missing = [n for n in names if n not in present]
    if missing:
        r.p(f"  NOT trained: {missing}")
    if len(pts) > 1:
        r.p(f"  ({len(pts)} .pt files present; the newest is what 01b loads)")


def sec_parcel_coverage(r):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cpc", Path(__file__).resolve().parent / "check_parcel_coverage.py")
    cpc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpc)
    cpc.main()


def _part_counts(root: Path, label: str, r):
    if not root.exists():
        r.p(f"  {label:<34} (absent)")
        return
    tables = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if not tables:
        r.p(f"  {label:<34} (present, empty)")
        return
    for t in tables:
        parts = list(t.rglob("*.parquet"))
        states = sorted({p.parent.name.replace("state=", "")
                         for p in parts if p.parent.name.startswith("state=")})
        r.p(f"  {label}/{t.name:<22} {len(parts):>5} part-file(s), "
            f"{len(states):>2} state(s)")
        if states:
            r.p(f"      {' '.join(states)}")


def sec_od_output(r):
    for root, label in ((C.OD_OUTPUT_DIR, "od_features"),
                        (C.OD_OUTPUT_DIR_CORRECTED, "od_features_corrected"),
                        (C.DATA_DIR / "od_features_candidates_train", "od_..._train"),
                        (C.DATA_DIR / "od_features_candidates", "od_..._candidates"),
                        (C.DATA_DIR / "od_features_candidates_queue", "od_..._queue")):
        _part_counts(root, label, r)


def sec_nlcd(r):
    if not C.NLCD_OUTPUT_DIR.exists():
        r.p("  (absent)")
        return
    files = sorted(C.NLCD_OUTPUT_DIR.glob(f"nlcd_*_k{C.K_RINGS}.parquet"))
    r.p(f"  {len(files)} state file(s) at k={C.K_RINGS}")
    r.p("  " + " ".join(sorted(f.stem.split("_")[1] for f in files)))


def sec_features(r):
    for name in ("05_plant_features.parquet", "10_parcel_features.parquet",
                 "14_stage1_training.parquet", "15_stage2_training.parquet",
                 "16_stage2b_training.parquet", "17_rerank_training.parquet",
                 "candidate_recall_failures.parquet"):
        p = C.FEATURES_OUTPUT_DIR / name
        if p.exists():
            st = p.stat()
            ts = dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="minutes")
            r.p(f"  OK      {name:<38} {st.st_size/1e6:>8.1f} MB  {ts}")
        else:
            r.p(f"  absent  {name}")
    shard = getattr(C, "FEATURE_SHARD_DIR", None)
    if shard and Path(shard).exists():
        ds = sorted(Path(shard).glob("state=*"))
        r.p(f"\n  feature_shards: {len(ds)} state dir(s)")
        if ds:
            r.p("  " + " ".join(d.name.replace("state=", "") for d in ds))


def sec_logs(r):
    if not C.LOGS_DIR.exists():
        r.p("  (no logs dir)")
        return
    logs = sorted(C.LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime,
                  reverse=True)
    r.p(f"  {len(logs)} log file(s); showing the newest "
        f"{min(LOG_FILES_PER_PATTERN * 2, len(logs))}\n")
    for f in logs[:LOG_FILES_PER_PATTERN * 2]:
        ts = dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="minutes")
        r.p(f"  --- {f.name}  ({f.stat().st_size/1024:.0f} KB, {ts}) ---")
        try:
            lines = f.read_text(errors="replace").splitlines()
        except Exception as e:
            r.p(f"      unreadable: {e}")
            continue
        for line in lines[-LOG_TAIL_LINES:]:
            r.p("      " + line[:200])
        r.p("")


def main():
    r = Report()
    r.p("tp_qa pipeline diagnostics")
    r.p("Generated by collect_diagnostics.py -- read-only snapshot.")

    r.section("1. ENVIRONMENT", sec_environment)
    r.section("2. CONFIGURED PATHS", sec_paths)
    r.section("3. DEPLOYED DETECTOR", sec_model)
    r.section("4. PARCEL COVERAGE vs TRAINING UNIVERSE", sec_parcel_coverage)
    r.section("5. 01a NLCD OUTPUT", sec_nlcd)
    r.section("6. DETECTION OUTPUT", sec_od_output)
    r.section("7. FEATURE TABLES", sec_features)
    r.section("8. RECENT JOB LOGS", sec_logs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = OUT_DIR / f"diag_{platform.node().split('.')[0]}_{stamp}.txt"
    out.write_text(r.buf.getvalue(), encoding="utf-8")

    print(r.buf.getvalue())
    print(f"\n{'=' * 78}")
    print(f"WRITTEN: {out}")
    print(f"  size: {out.stat().st_size/1024:.0f} KB")
    print("\nCommit and push it:")
    print(f"  git add {out.relative_to(C.ROOT.parent)}")
    print(f'  git commit -m "diagnostics {stamp}"')
    print(f"  git push")


if __name__ == "__main__":
    main()
