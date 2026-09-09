"""
diagnose_od_class_shift.py
==========================
Read-only. Answers one question before a national run with new object
detection weights: how much of Stage 1 / 2 / 2b's predictive signal comes
from OD classes the new model can no longer emit?

WHY
    detection/'s 04_train_model.py has KEEP_CLASSES set to 3 classes
    (aeration_basin, clarifier, digester). C.CLASSES here still lists 6, and
    that is CORRECT -- 01b resolves class NAMES from model.names at runtime
    (01b:753) and iterates C.CLASSES by name when building features
    (01b:495), so a 3-class model produces a stable 6-class feature schema
    with zeros in the gaps. Nothing breaks and nothing is mislabeled.

    But Stage 1/2/2b were TRAINED when those columns held real values. Under
    the new weights:
        od_has_{dead}       True/False  -> always False
        od_n_{dead}         counts      -> always 0
        od_max_conf_{dead}  0..1        -> NaN, then filled to 0 (02:629)
        od_dominant_class   6 values    -> only 3 reachable
    The models still run. Their scores just shift, silently, in proportion
    to how much they leaned on those columns.

    This script measures that proportion. It does not change anything.

READING THE RESULT
    Dead-share under ~1%   -> noise, proceed.
    Dead-share ~1-5%       -> ranking will shift at the margin; fine for a
                              survey run, note it when comparing to the
                              round that produced the current review set.
    Dead-share above ~5%   -> Stage 2a's ordering is materially different
                              from the one your last review round was based
                              on. Retraining the stages on features
                              regenerated with the new weights is the clean
                              fix; that is a separate job, not this one.

Usage (via sbatch diagnose_od_class_shift.slurm):
    python diagnose_od_class_shift.py
    python diagnose_od_class_shift.py --weights /path/to/best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

IMPORTANCE_FILES = {
    "stage1":  "stage1_rf_importance.parquet",
    "stage2":  "stage2_rf_importance.parquet",
    "stage2b": "stage2b_rf_importance.parquet",
}

PER_CLASS_PREFIXES = ["od_has_", "od_n_", "od_max_conf_"]


def resolve_weights(explicit) -> Path | None:
    """Newest best.pt under OD_MODEL_DIR, mirroring how 01b picks weights."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    cands = sorted(C.OD_MODEL_DIR.rglob("*.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def model_class_names(weights: Path) -> list[str] | None:
    """Names straight from the checkpoint -- same source 01b:753 uses."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  WARNING: ultralytics not importable; cannot read model.names.")
        return None
    try:
        m = YOLO(str(weights))
        return [m.names[k] for k in sorted(m.names)]
    except Exception as e:
        print(f"  WARNING: could not load {weights.name}: {e}")
        return None


def load_importance(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Column naming varies across 03/04/07; find the feature and score cols
    feat_col = next((c for c in ("feature", "Feature", "feature_name", "name")
                     if c in df.columns), df.columns[0])
    imp_col = next((c for c in ("importance", "Importance", "mean_importance",
                                "gain", "value") if c in df.columns),
                   df.columns[1] if len(df.columns) > 1 else None)
    if imp_col is None:
        return None
    out = df[[feat_col, imp_col]].copy()
    out.columns = ["feature", "importance"]
    out["importance"] = pd.to_numeric(out["importance"], errors="coerce")
    return out.dropna(subset=["importance"])


def classify_feature(name: str, live: set, dead: set) -> str:
    for pref in PER_CLASS_PREFIXES:
        if name.startswith(pref):
            cls = name[len(pref):]
            if cls in dead:
                return "dead"
            if cls in live:
                return "live"
            # Not a per-class column at all. 01b:495 only emits these for
            # names in C.CLASSES, so a non-matching suffix means an aggregate
            # that happens to share the prefix -- od_n_objects being the one
            # that actually occurs. Fall through rather than inventing a
            # class called "objects".
            break
    if name.startswith("od_dominant_class"):
        # One-hot expansions look like od_dominant_class_clarifier
        suffix = name[len("od_dominant_class"):].lstrip("_")
        if suffix in dead:
            return "dead"
        return "live" if suffix else "categorical"
    return "other"


def report_stage(stage: str, imp: pd.DataFrame, live: set, dead: set):
    imp = imp.copy()
    imp["kind"] = [classify_feature(f, live, dead) for f in imp["feature"]]
    total = imp["importance"].sum()
    if total <= 0:
        print(f"  {stage}: importances sum to {total}; cannot compute shares.")
        return None

    imp["share"] = imp["importance"] / total
    imp["rank"] = imp["importance"].rank(ascending=False, method="min").astype(int)

    by_kind = imp.groupby("kind")["share"].sum().sort_values(ascending=False)
    dead_share = float(by_kind.get("dead", 0.0))

    print(f"\n  --- {stage} ({len(imp)} features) ---")
    for kind, share in by_kind.items():
        print(f"    {kind:14s} {share*100:6.2f}% of total importance")

    dead_rows = imp[imp["kind"] == "dead"].sort_values("importance", ascending=False)
    if len(dead_rows):
        print(f"    dead features, highest first:")
        for r in dead_rows.head(10).itertuples():
            print(f"      rank {r.rank:>3}  {r.share*100:5.2f}%  {r.feature}")
        if len(dead_rows) > 10:
            print(f"      ... and {len(dead_rows) - 10} more")

    top = imp.nsmallest(15, "rank")
    n_dead_top = int((top["kind"] == "dead").sum())
    print(f"    dead features inside the top 15: {n_dead_top}")
    return dead_share


def report_dominant_class(live: set, dead: set):
    """od_dominant_class is categorical. Categories that were common in
    training and are now unreachable mean dead one-hot columns, which the
    importance table may not expose separately."""
    path = C.FEATURES_OUTPUT_DIR / "14_stage1_training.parquet"
    if not path.exists():
        print(f"\n  (skipping od_dominant_class check -- {path.name} not found)")
        return
    try:
        df = pd.read_parquet(path, columns=["od_dominant_class"])
    except Exception as e:
        print(f"\n  (skipping od_dominant_class check -- {e})")
        return

    vc = df["od_dominant_class"].value_counts(dropna=False)
    n = int(vc.sum())
    print(f"\n  --- od_dominant_class in training features ({n} rows) ---")
    unreachable = 0
    for val, k in vc.items():
        tag = ""
        if isinstance(val, str) and val in dead:
            tag = "  <-- UNREACHABLE with new weights"
            unreachable += k
        print(f"    {str(val):20s} {k:7d}  {k/n*100:5.2f}%{tag}")
    if unreachable:
        print(f"    {unreachable} row(s) ({unreachable/n*100:.2f}%) had a dominant "
              f"class the new model cannot produce; those parcels will fall to "
              f"a different dominant class or none.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="explicit best.pt (default: newest in OD_MODEL_DIR)")
    args = ap.parse_args()

    print("=== diagnose_od_class_shift.py ===")
    print(f"MODELS_DIR   : {C.MODELS_DIR}")
    print(f"OD_MODEL_DIR : {C.OD_MODEL_DIR}\n")

    weights = resolve_weights(args.weights)
    if weights is None:
        print(f"ERROR: no .pt found under {C.OD_MODEL_DIR}. Upload best.pt first.")
        sys.exit(1)
    print(f"Weights: {weights}")
    print(f"  modified: {pd.Timestamp(weights.stat().st_mtime, unit='s')}")

    names = model_class_names(weights)
    if names is None:
        print("\nCannot read model classes -- aborting rather than guessing.")
        sys.exit(1)

    live = set(names)
    configured = list(C.CLASSES)
    dead = set(configured) - live
    extra = live - set(configured)

    print(f"\nModel emits {len(names)} class(es): {names}")
    print(f"config.CLASSES lists {len(configured)}: {configured}")
    print(f"  live (model can emit) : {sorted(live)}")
    print(f"  dead (schema only)    : {sorted(dead) if dead else 'none'}")
    if extra:
        print(f"  WARNING: model emits {sorted(extra)}, absent from C.CLASSES. "
              f"01b:495 iterates C.CLASSES, so these detections are counted in "
              f"od_n_objects but get NO per-class feature. Add them to "
              f"C.CLASSES before running.")
    if not dead:
        print("\nNo dead classes -- feature schema fully exercised. Nothing to weigh.")
        return

    print("\nPer-stage importance attributable to dead classes:")
    shares = {}
    for stage, fname in IMPORTANCE_FILES.items():
        imp = load_importance(C.MODELS_DIR / fname)
        if imp is None:
            print(f"\n  --- {stage} --- not found ({fname}); skipped")
            continue
        s = report_stage(stage, imp, live, dead)
        if s is not None:
            shares[stage] = s

    report_dominant_class(live, dead)

    print("\n=== verdict ===")
    if not shares:
        print("  No importance tables readable -- cannot judge. Check "
              f"{C.MODELS_DIR} for *_rf_importance.parquet.")
        return
    worst = max(shares.values())
    for stage, s in shares.items():
        print(f"  {stage:8s} dead-class share: {s*100:5.2f}%")
    print()
    if worst < 0.01:
        print("  Under 1%. Noise. The national run is safe to launch.")
    elif worst < 0.05:
        print("  1-5%. Scores will shift slightly. Fine for a survey run, but "
              "don't compare rankings directly against the round that produced "
              "your current review set.")
    else:
        print("  Above 5%. Stage 2a's ordering will differ materially from the "
              "run your review set came from. The run will complete and look "
              "normal -- the difference is invisible in the logs. Consider "
              "regenerating features with the new weights and retraining the "
              "stages before treating the output as comparable.")


if __name__ == "__main__":
    main()
