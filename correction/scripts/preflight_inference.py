"""
preflight_inference.py
======================
Read-only check that everything 05_run_inference_array needs is in place,
BEFORE the array is submitted. Writes one text report to
correction/diagnostics/ so it can be pushed back and read elsewhere.

    sbatch preflight_inference.slurm
    -> correction/diagnostics/preflight_inference_<host>_<YYYYMMDD-HHMMSS>.txt

WHY THIS EXISTS (2026-09-24)
    05 scores the FULL treatment-plant universe (~16k plants), but it does not
    build any features of its own -- it reads what 02 and 01a/01b left behind:

      05_plant_features.parquet / 10_parcel_features.parquet
          02 overwrites these on every run. Run WITHOUT FULL_UNIVERSE=1 (the
          normal training run), they cover only the ~2.6k labelled plants.
          05 does not fail on that: its merges are left joins, so every
          unlabelled plant is scored on empty features and the queue fills
          with garbage that looks like model output.

      nlcd_{STATE}_k18.parquet (01a)
          Stage 2a's candidate pool for a plant is the parcels in this file
          within K_RINGS of its reported point. A training-universe file only
          holds parcels near labelled plants, so unlabelled plants elsewhere
          get few or no candidates -- again silently.

      od_features/plants (01b)
          Stage 1's od_* features at the REPORTED location. A plant with no
          row scores as od_ran=False, which is not what the model trained on
          for most labelled plants.

    And the re-ranker (07b) takes stage2a_rank / stage2_prob_correct as
    features, so a re-ranker trained before the current Stage 2a is scoring
    against a different model's output.

    None of these errors. All of them are cheap to check first.

Every section is read-only and isolated: one failing prints its traceback in
place and the rest still run. The verdict block at the end lists what to fix.
"""
import argparse
import datetime as dt
import platform
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from collect_diagnostics import Report, sec_environment

OUT_DIR = C.ROOT / "diagnostics"

# A state whose coverage falls below this is flagged. Deliberately loose:
# the point is to tell "built for the full universe" (~100%) apart from
# "built for training only" (typically a small fraction), not to chase the
# last few plants with unusable coordinates.
COVERAGE_WARN = 0.90

PROBLEMS: list[str] = []


def flag(msg: str):
    PROBLEMS.append(msg)
    print(f"  !! {msg}")


def fmt_time(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes")


def load_universe(states: list[str]) -> pd.DataFrame:
    """Same plant set 05_run_inference.load_full_universe_plants builds."""
    import h3

    ft = pd.read_csv(C.CWNS_DIR / "FACILITY_TYPES.txt", dtype=str, encoding="latin1")
    treatment = set(ft.loc[ft["FACILITY_TYPE"] == "Treatment Plant", "CWNS_ID"])
    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt",
                      dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc[loc["CWNS_ID"].isin(treatment)].copy()
    loc["LATITUDE"] = pd.to_numeric(loc["LATITUDE"], errors="coerce")
    loc["LONGITUDE"] = pd.to_numeric(loc["LONGITUDE"], errors="coerce")
    loc = loc.dropna(subset=["LATITUDE", "LONGITUDE"]).drop_duplicates(subset="CWNS_ID")
    loc = loc[loc["STATE_CODE"].isin(states)].reset_index(drop=True)
    loc["h3_res9"] = [h3.latlng_to_cell(la, lo, 9)
                      for la, lo in zip(loc["LATITUDE"], loc["LONGITUDE"])]
    return loc[["CWNS_ID", "STATE_CODE", "h3_res9"]]


def per_state_coverage(universe: pd.DataFrame, covered: set, label: str):
    """Print coverage per state and flag the ones below COVERAGE_WARN."""
    u = universe.assign(hit=universe["CWNS_ID"].isin(covered))
    tot = u["hit"].mean() if len(u) else 0.0
    print(f"  {label}: {int(u['hit'].sum())}/{len(u)} universe plants ({tot:.1%})")
    by = u.groupby("STATE_CODE")["hit"].agg(["sum", "size"])
    by["share"] = by["sum"] / by["size"]
    low = by[by["share"] < COVERAGE_WARN].sort_values("share")
    if len(low):
        print(f"  {len(low)} state(s) below {COVERAGE_WARN:.0%}:")
        for st, r in low.iterrows():
            print(f"    {st}: {int(r['sum'])}/{int(r['size'])} ({r['share']:.0%})")
    return tot, low


def sec_universe(r, ctx):
    u = load_universe(ctx["states"])
    ctx["universe"] = u
    print(f"  states requested : {len(ctx['states'])}")
    print(f"  treatment plants with usable coordinates: {len(u)}")
    missing = sorted(set(ctx["states"]) - set(u["STATE_CODE"]))
    if missing:
        print(f"  states with no plants at all: {missing}")


def sec_plant_features(r, ctx):
    p = C.FEATURES_OUTPUT_DIR / "05_plant_features.parquet"
    if not p.exists():
        flag("05_plant_features.parquet is missing -- run 02 with FULL_UNIVERSE=1 + 02b")
        return
    pf = pd.read_parquet(p, columns=["CWNS_ID", "STATE_CODE"])
    pf["CWNS_ID"] = pf["CWNS_ID"].astype(str)
    print(f"  {p.name}: {len(pf)} rows, written {fmt_time(p)}")
    tot, low = per_state_coverage(ctx["universe"], set(pf["CWNS_ID"]),
                                  "plants with plant-level features")
    if tot < COVERAGE_WARN:
        flag(f"05_plant_features covers only {tot:.0%} of the universe -- it looks "
             f"like a TRAINING-universe build. Re-run 02 with FULL_UNIVERSE=1, "
             f"then 02b, before 05.")


def sec_parcel_features(r, ctx):
    import pyarrow.parquet as pq

    p = C.FEATURES_OUTPUT_DIR / "10_parcel_features.parquet"
    if not p.exists():
        flag("10_parcel_features.parquet is missing -- run 02 + 02b")
        return
    n = pq.ParquetFile(p).metadata.num_rows
    print(f"  {p.name}: {n:,} rows, written {fmt_time(p)}")
    ctx["parcel_features_rows"] = n

    # Row count of 01a's output for the same states. 02 builds 10_ from those
    # files, so the two should be the same order of magnitude; a 10_ far
    # smaller than 01a's output means 02 ran on a narrower scope.
    total_nlcd = 0
    for st in ctx["states"]:
        f = C.NLCD_OUTPUT_DIR / f"nlcd_{st}_k{C.K_RINGS}.parquet"
        if f.exists():
            total_nlcd += pq.ParquetFile(f).metadata.num_rows
    print(f"  01a NLCD rows across the same states: {total_nlcd:,}")
    if total_nlcd and n < 0.9 * total_nlcd:
        flag(f"10_parcel_features has {n:,} rows against {total_nlcd:,} in 01a's "
             f"output -- 02 was run on a narrower scope than 01a. Re-run 02 with "
             f"FULL_UNIVERSE=1 + 02b.")


def sec_nlcd(r, ctx):
    """Does each state's 01a file cover the whole universe, or only the
    neighbourhoods of labelled plants?

    Test: is the parcel set in nlcd_{STATE} non-empty in the plant's OWN
    reported H3 cell or its immediate neighbours (k=1, ~0.5 km)? A
    full-universe build searched a k=18 disk around every plant, so almost
    every plant passes. A training-only build misses most unlabelled plants
    that are not near a labelled one. Plants on no parcel at all (water,
    Regrid gaps) fail both ways, so expect a little under 100% even when
    the build is right.
    """
    import h3

    u = ctx["universe"]
    covered = set()
    missing_files = []
    for st, grp in u.groupby("STATE_CODE"):
        f = C.NLCD_OUTPUT_DIR / f"nlcd_{st}_k{C.K_RINGS}.parquet"
        if not f.exists():
            missing_files.append(st)
            continue
        cells = set(pd.read_parquet(f, columns=["h3_index_9"])["h3_index_9"])
        for cid, c in zip(grp["CWNS_ID"], grp["h3_res9"]):
            if any(n in cells for n in h3.grid_disk(c, 1)):
                covered.add(cid)
        del cells
    if missing_files:
        flag(f"no 01a output for: {' '.join(missing_files)}")
    tot, low = per_state_coverage(u, covered, "plants with 01a parcels at their reported point")
    if tot < 0.75:
        flag(f"01a output reaches only {tot:.0%} of plants' reported points -- "
             f"looks like a training-universe build. Re-run 01a with "
             f"FULLUNIVERSE=1 NORESUME=1 for the flagged states (then 01d, "
             f"then 02 FULL_UNIVERSE=1).")


def sec_od_reported(r, ctx):
    plants_dir = C.OD_OUTPUT_DIR / "plants"
    if not plants_dir.exists():
        flag("no 01b output (od_features/plants) -- Stage 1 would score with no OD")
        return
    ids = set()
    for f in plants_dir.rglob("part-*.parquet"):
        try:
            ids |= set(pd.read_parquet(f, columns=["CWNS_ID"])["CWNS_ID"].astype(str))
        except Exception as e:
            print(f"  could not read {f}: {e}")
    tot, low = per_state_coverage(ctx["universe"], ids, "plants with a 01b result")
    if tot < COVERAGE_WARN:
        flag(f"01b has run for only {tot:.0%} of the universe. Stage 1 will score the "
             f"rest with od_ran=False. Run 01b full-universe for the listed states "
             f"(it resumes per plant), or accept it knowingly.")


def sec_freshness(r, ctx):
    """check_od_freshness in report-only mode. It signals through its exit
    code (0 clean, 1 stale, 2 setup error), so catch that rather than let it
    end this job."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cof", Path(__file__).resolve().parent / "check_od_freshness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    argv, sys.argv = sys.argv, ["check_od_freshness"]
    try:
        mod.main()
        code = 0
    except SystemExit as e:
        code = e.code or 0
    finally:
        sys.argv = argv
    if code == 1:
        flag("check_od_freshness found detection partitions older than best.pt "
             "(see the state list above)")
    elif code == 2:
        flag("check_od_freshness could not run (setup error above)")


def sec_models(r, ctx):
    names = {
        "best.pt": None,
        "stage1_rf_model.joblib": C.MODELS_DIR / "stage1_rf_model.joblib",
        "stage2_rf_model.joblib": C.MODELS_DIR / "stage2_rf_model.joblib",
        "rerank_rf_model.joblib": C.MODELS_DIR / "rerank_rf_model.joblib",
        "rerank_rf_model_noOD.joblib": C.MODELS_DIR / "rerank_rf_model_noOD.joblib",
        "05_plant_features.parquet": C.FEATURES_OUTPUT_DIR / "05_plant_features.parquet",
        "14_stage1_training.parquet": C.FEATURES_OUTPUT_DIR / "14_stage1_training.parquet",
        "15_stage2_training.parquet": C.FEATURES_OUTPUT_DIR / "15_stage2_training.parquet",
        "17_rerank_training.parquet": C.FEATURES_OUTPUT_DIR / "17_rerank_training.parquet",
        "inference/stage2_candidates.parquet": C.DATA_DIR / "inference" / "stage2_candidates.parquet",
        "inference/stage2_candidates_reranked.parquet":
            C.DATA_DIR / "inference" / "stage2_candidates_reranked.parquet",
    }
    pts = sorted(C.OD_MODEL_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime,
                 reverse=True) if C.OD_MODEL_DIR.exists() else []
    names["best.pt"] = pts[0] if pts else None

    times = {}
    for label, p in names.items():
        if p is None or not Path(p).exists():
            print(f"  absent   {label}")
            continue
        times[label] = Path(p).stat().st_mtime
        print(f"  {fmt_time(Path(p))}  {label}")

    s2 = times.get("stage2_rf_model.joblib")
    rr = times.get("rerank_rf_model.joblib")
    if s2 and rr and rr < s2:
        flag("the re-ranker is OLDER than the current Stage 2a model. It uses Stage "
             "2a's rank and score as features, so retrain it after 05 + merge: "
             "01e SCOPE=train -> 06b -> 07b, then 05b.")
    s1 = times.get("stage1_rf_model.joblib")
    pf = times.get("05_plant_features.parquet")
    if s1 and pf and pf > s1:
        print("  note: 05_plant_features is newer than Stage 1 -- expected if 02 was "
              "re-run FULL_UNIVERSE=1 after training. Training tables are unchanged "
              "by that, so no retrain is needed.")


def sec_shards(r, ctx):
    root = C.DATA_DIR / "inference_shards"
    if not root.exists():
        print("  no inference_shards/ yet -- expected before the first array run")
        return
    dirs = sorted(root.glob("state=*"))
    print(f"  {len(dirs)} existing shard dir(s). The array overwrites each state's "
          f"shard; merge_05_shards refuses if a requested state is missing.")
    s1 = C.MODELS_DIR / "stage1_rf_model.joblib"
    if s1.exists():
        old = [d.name.replace("state=", "") for d in dirs
               if (d / "plant_summary.parquet").exists()
               and (d / "plant_summary.parquet").stat().st_mtime < s1.stat().st_mtime]
        if old:
            print(f"  {len(old)} shard(s) predate the current Stage 1 model (will be "
                  f"replaced when the array runs): {' '.join(old)}")


def sec_holdout(r, ctx):
    m = C.DATA_DIR / "holdout" / "holdout_manifest.parquet"
    t = C.DATA_DIR / "holdout" / "holdout_truth.parquet"
    if not m.exists():
        flag("no holdout manifest -- 12_score_holdout cannot run")
        return
    man = pd.read_parquet(m)
    man["CWNS_ID"] = man["CWNS_ID"].astype(str)
    truth = pd.read_parquet(t) if t.exists() else pd.DataFrame(columns=["CWNS_ID"])
    truth_ids = set(truth["CWNS_ID"].astype(str))
    print(f"  manifest: {len(man)} plants, cohorts {sorted(man['cohort'].unique())}")
    for b, g in man.groupby("bin"):
        n_t = int(g["CWNS_ID"].isin(truth_ids).sum())
        print(f"    {b:12s} {len(g):4d} plants, {n_t:4d} with truth")
    un = man[man["bin"] == "unlabeled"]
    if len(un) and un["CWNS_ID"].isin(truth_ids).mean() < 0.5:
        print("  note: most unlabeled-bin holdout plants still have no truth. 10 only "
              "queues them in round 1, so the deployment-distribution number stays "
              "empty until that changes. Not a blocker for this run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True,
                    help="space- or comma-separated; the wrapper passes DEFAULT_STATES")
    args = ap.parse_args()
    ctx = {"states": [s for s in args.states.replace(",", " ").split() if s]}

    r = Report()
    r.p("tp_qa inference preflight")
    r.p("Generated by preflight_inference.py -- read-only.")

    r.section("1. ENVIRONMENT", sec_environment)
    r.section("2. PLANT UNIVERSE (what 05 will score)", lambda r: sec_universe(r, ctx))
    if "universe" in ctx:
        r.section("3. PLANT FEATURES SCOPE (02)", lambda r: sec_plant_features(r, ctx))
        r.section("4. PARCEL FEATURES SCOPE (02)", lambda r: sec_parcel_features(r, ctx))
        r.section("5. CANDIDATE PARCELS SCOPE (01a)", lambda r: sec_nlcd(r, ctx))
        r.section("6. OD AT REPORTED LOCATIONS (01b)", lambda r: sec_od_reported(r, ctx))
    else:
        PROBLEMS.append("could not build the plant universe -- sections 3-6 skipped")
    r.section("7. DETECTION FRESHNESS vs best.pt", lambda r: sec_freshness(r, ctx))
    r.section("8. MODEL AND TABLE TIMESTAMPS", lambda r: sec_models(r, ctx))
    r.section("9. EXISTING INFERENCE SHARDS", lambda r: sec_shards(r, ctx))
    r.section("10. HOLDOUT", lambda r: sec_holdout(r, ctx))

    r.h("VERDICT")
    if PROBLEMS:
        r.p(f"{len(PROBLEMS)} thing(s) to deal with before or during this run:")
        for i, p in enumerate(PROBLEMS, 1):
            r.p(f"  {i}. {p}")
    else:
        r.p("Nothing flagged. Safe to submit 05_run_inference_array.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = OUT_DIR / f"preflight_inference_{platform.node().split('.')[0]}_{stamp}.txt"
    out.write_text(r.buf.getvalue(), encoding="utf-8")

    print(r.buf.getvalue())
    print(f"\nWRITTEN: {out}")
    print("\nCommit and push it, from a login node:")
    print("  cd /work/GRDVULN/tp_qa")
    print("  git add correction/diagnostics/")
    print(f'  git commit -m "preflight {stamp}"')
    print("  git push")


if __name__ == "__main__":
    main()
