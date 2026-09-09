"""
06b_build_rerank_training.py
============================
Builds the training table for a Stage 2a RE-RANKER: one row per candidate
parcel, labelled 1 if it is the plant's true parcel and 0 if it is one of
the 19 competitors that Stage 2a also surfaced.

WHY THIS IS NOT 06_build_stage2b_training.py
    06 pairs the REPORTED parcel (label 0) against the CORRECTED parcel
    (label 1). Those two are usually nothing alike -- a residential lot
    versus a 40-acre municipal parcel on a river -- so parcel attributes
    separate them on their own and OD has no residual variance to explain.
    That is why Stage 2b's measured OD importance is near zero, and it is a
    fact about the training data rather than about object detection.

    Measured directly on the holdout (diagnose_candidate_od.py, 2026-09-08):
    among Stage 2a's top-20 candidates, true parcels fire OD at 46.4% and
    their competitors at 9.0% -- a 5x ratio, with od_n_objects 2.64 vs 0.32.
    Ranking by OD confidence alone picks the true parcel 35.7% of the time
    against a 5% random baseline. The signal is there; 06's training pairs
    simply could not show it.

    This script builds the distribution 05_run_inference.py's docstring
    calls for when it defers the re-rank step pending "distribution-matched
    training data".

WHAT A ROW IS
    (CWNS_ID, ll_uuid) from stage2_candidates.parquet, restricted to
    corrections-layer plants NOT in the holdout, joined to:
      - the parcel features Stage 2a itself scored on
      - OD features from 01e --training-corrections
      - stage2_prob_correct and its within-plant rank

    stage2_prob_correct is deliberately INCLUDED as a feature. The re-ranker
    should be able to fall back on Stage 2a's own judgement when OD says
    nothing -- 16 of 44 holdout plants had zero detections across all 20
    candidates, and for those the OD columns are constant and uninformative.
    Learning that fallback beats hard-coding it, though 05b keeps a hard
    fallback as a backstop.

PLANTS WHOSE TRUE PARCEL IS NOT IN THE POOL
    Kept, with 20 negatives and no positive, and flagged via
    `pool_has_positive`. They are real: they teach what a near-miss looks
    like, and dropping them would train the model on an easier world than
    the one it is deployed into. 07b can weight or exclude them from the
    flag. 08 (2026-09-08) puts the ceiling at 91% of corrections having
    their true parcel inside the k=18 ring at all; the holdout measured 64%
    reaching the top 20.

Output: data/features/17_rerank_training.parquet

Usage:
    python 06b_build_rerank_training.py --dry-run
    python 06b_build_rerank_training.py
    python 06b_build_rerank_training.py --top-k 20
"""
import argparse
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from holdout import exclude_holdout

OD_TRAIN_ROOT = C.DATA_DIR / "od_features_candidates_train" / "candidates"
OUT_PATH = C.FEATURES_OUTPUT_DIR / "17_rerank_training.parquet"


# ===========================================================================
def load_true_parcels(con) -> pd.DataFrame:
    """Resolve each correction's corrected point to its containing parcel.

    Same derivation 09_build_holdout.py's lookup_true_parcels uses, and the
    same one 06 relies on for its positive rows -- a correction's truth is
    the parcel containing Corrected_X/Y. Reimplemented here rather than
    imported because 09's version is bound to its manifest/truth-file
    plumbing, which does not apply to training plants.
    """
    corr = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CORRECTIONS)
    corr["CWNS_ID"] = corr["CWNS_ID"].astype(str)
    corr["true_lon"] = pd.to_numeric(corr["Corrected_X"], errors="coerce")
    corr["true_lat"] = pd.to_numeric(corr["Corrected_Y"], errors="coerce")
    corr = corr.dropna(subset=["true_lon", "true_lat"])

    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt",
                      dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc.drop_duplicates(subset="CWNS_ID")
    corr = corr.merge(loc[["CWNS_ID", "STATE_CODE"]], on="CWNS_ID", how="left")
    corr = corr.dropna(subset=["STATE_CODE"])
    print(f"  Corrections with usable coordinates: {len(corr)}")

    frames = []
    for state, grp in corr.groupby("STATE_CODE"):
        pts = grp[["CWNS_ID", "true_lon", "true_lat"]].rename(
            columns={"true_lon": "LON", "true_lat": "LAT"})
        con.register("pts", pts)
        try:
            res = con.execute(f"""
                SELECT pts.CWNS_ID, p.{C.PARCEL_ID_FIELD} AS true_ll_uuid
                FROM read_parquet('{C.PARCEL_BASE.as_posix()}/state={state}/*.parquet') p
                JOIN pts ON ST_Intersects(
                    ST_GeomFromWKB(p.{C.PARCEL_WKB_FIELD}), ST_Point(pts.LON, pts.LAT))
            """).df()
            if len(res):
                res["CWNS_ID"] = res["CWNS_ID"].astype(str)
                frames.append(res.drop_duplicates(subset="CWNS_ID"))
        except Exception as e:
            print(f"    [{state}] true-parcel lookup failed: {str(e)[:160]}")
        finally:
            con.unregister("pts")

    truth = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=["CWNS_ID", "true_ll_uuid"])
    truth["true_ll_uuid"] = truth["true_ll_uuid"].astype(str)
    print(f"  True parcel resolved for {len(truth)}/{len(corr)} correction(s)")
    return truth


def load_candidate_od() -> pd.DataFrame:
    """01e --training-corrections output. Deliberately reads ONLY the
    training root: od_features_candidates/ holds the holdout's rows, and
    mixing them would train a model on its own evaluation set."""
    if not OD_TRAIN_ROOT.exists() or not any(OD_TRAIN_ROOT.rglob("*.parquet")):
        print(f"ERROR: no OD output under {OD_TRAIN_ROOT}.\n"
              f"  Run: sbatch --export=SCOPE=\"train\" 01e_run_od_candidates.slurm")
        sys.exit(2)
    files = sorted(OD_TRAIN_ROOT.rglob("*.parquet"),
                   key=lambda f: f.stat().st_mtime)
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        for col in df.select_dtypes(include=["category"]).columns:
            df[col] = df[col].astype(str)
        frames.append(df)
    od = pd.concat(frames, ignore_index=True)
    od["CWNS_ID"] = od["CWNS_ID"].astype(str)
    od["ll_uuid"] = od["ll_uuid"].astype(str)
    # mtime-sorted keep="last", matching 02's dedup of 01b's append-only output
    od = od.drop_duplicates(subset=["CWNS_ID", "ll_uuid"], keep="last")
    return od.drop(columns=["state", "orig_lon", "orig_lat", "n_objects_total",
                            "n_objects_in_parcel", "processed_at"], errors="ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=20,
                    help="candidates per plant (must match what 01e fetched)")
    ap.add_argument("--drop-no-positive", action="store_true",
                    help="exclude plants whose true parcel is not in the pool. "
                         "Off by default -- those plants are real and their "
                         "negatives teach what a near-miss looks like.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-no-holdout", action="store_true")
    args = ap.parse_args()

    print("=== 06b_build_rerank_training.py ===\n")

    cand_path = C.DATA_DIR / "inference" / "stage2_candidates.parquet"
    if not cand_path.exists():
        print(f"ERROR: {cand_path} not found. Run 05 (or the array + merge) first.")
        sys.exit(2)

    print("Loading Stage 2a candidates...")
    cands = pd.read_parquet(cand_path)
    cands["CWNS_ID"] = cands["CWNS_ID"].astype(str)
    cands["ll_uuid"] = cands["ll_uuid"].astype(str)
    cands = cands.sort_values(["CWNS_ID", "stage2_prob_correct"],
                              ascending=[True, False])
    cands = cands.groupby("CWNS_ID", group_keys=False).head(args.top_k)
    # Within-plant rank. The re-ranker sees WHERE Stage 2a put a candidate,
    # not just its probability -- probabilities are poorly calibrated across
    # plants (a 0.4 in a weak pool may outrank a 0.7 in a strong one), while
    # rank is meaningful within the competition being resolved.
    cands["stage2a_rank"] = cands.groupby("CWNS_ID").cumcount() + 1
    print(f"  {len(cands)} candidate(s) across {cands['CWNS_ID'].nunique()} plant(s)")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")
    print("\nResolving true parcels for corrections...")
    truth = load_true_parcels(con)
    con.close()

    print("\nLoading candidate OD features (01e --training-corrections)...")
    od = load_candidate_od()
    print(f"  {len(od)} candidate(s) with OD output "
          f"across {od['CWNS_ID'].nunique()} plant(s)")

    # ---- restrict to plants that can actually contribute -----------------
    train = cands[cands["CWNS_ID"].isin(set(truth["CWNS_ID"]))].copy()
    print(f"\nCandidates for corrections plants: {len(train)} "
          f"across {train['CWNS_ID'].nunique()} plant(s)")

    train = exclude_holdout(train, "rerank-build",
                            allow_missing=args.allow_no_holdout)

    with_od = set(zip(od["CWNS_ID"], od["ll_uuid"]))
    has_od = [(c, u) in with_od for c, u in zip(train["CWNS_ID"], train["ll_uuid"])]
    n_no_od = len(train) - sum(has_od)
    if n_no_od:
        print(f"  {n_no_od} candidate(s) have no OD row -- 01e skipped them "
              f"(no NAIP coverage, or no parcel geometry). Kept, with OD "
              f"columns defaulted below; od_ran=False distinguishes them "
              f"from parcels where detection genuinely found nothing.")

    # ---- label ------------------------------------------------------------
    train = train.merge(truth, on="CWNS_ID", how="left")
    train["label"] = (train["ll_uuid"] == train["true_ll_uuid"]).astype(int)

    pos_per_plant = train.groupby("CWNS_ID")["label"].max()
    n_with_pos = int(pos_per_plant.sum())
    n_plants = len(pos_per_plant)
    print(f"\n  True parcel present in the top-{args.top_k}: "
          f"{n_with_pos}/{n_plants} plant(s) ({n_with_pos / n_plants:.0%} "
          f"candidate recall@{args.top_k})")
    train["pool_has_positive"] = train["CWNS_ID"].map(pos_per_plant).astype(bool)

    if args.drop_no_positive:
        before = len(train)
        train = train[train["pool_has_positive"]]
        print(f"  --drop-no-positive: {before} -> {len(train)} rows")

    # ---- OD join ----------------------------------------------------------
    train = train.merge(od, on=["CWNS_ID", "ll_uuid"], how="left")

    # Same fillna contract 02 and 06 use for 01b's output: a missing OD row
    # is "nothing detected", not "unknown". od_max_conf_* is left NaN on
    # purpose -- there is no confidence when there is no detection, and the
    # imputer handles it (07 already drops all-NaN OD columns).
    if "od_ran" in train.columns:
        train["od_ran"] = train["od_ran"].fillna(False)
    if "od_has_detection" in train.columns:
        train["od_has_detection"] = train["od_has_detection"].fillna(False)
    for c in train.columns:
        if c.startswith("od_has_"):
            train[c] = train[c].fillna(False)
        elif c.startswith("od_n_"):
            train[c] = train[c].fillna(0)

    # ---- pool-level OD context -------------------------------------------
    # Whether ANY candidate in this plant's pool fired. When none did, every
    # OD column is constant within the competition and carries no ranking
    # information -- the model should lean on stage2_prob_correct instead.
    # Giving it the flag lets it learn that rather than being told.
    fired = train.groupby("CWNS_ID")["od_has_detection"].transform("sum")
    train["od_pool_n_fired"] = fired.astype(int)
    train["od_any_fired_in_pool"] = fired > 0
    # How unusual this candidate's detection is WITHIN its own pool. A
    # detection is far more telling when it is the only one among 20 than
    # when 18 of 20 fired.
    train["od_fired_share_of_pool"] = np.where(
        fired > 0, train["od_has_detection"].astype(float) / fired, 0.0)

    # ---- spatial CV coordinates ------------------------------------------
    # From the CANDIDATE parcel's own location, not the plant's reported
    # point: each row IS a candidate, and 07b's spatial folds cluster on
    # per-plant means of these.
    if {"centroid_lng", "centroid_lat"}.issubset(train.columns):
        lon, lat = train["centroid_lng"], train["centroid_lat"]
    else:
        lon, lat = train["LONGITUDE"], train["LATITUDE"]
        print("  NOTE: no candidate centroid columns -- using the plant's "
              "reported point for x_5070/y_5070. Spatial folds will cluster "
              "by plant rather than by candidate, which is the intended "
              "grouping anyway.")
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat),
                        crs=C.EXPORT_CRS).to_crs(C.PROJECTED_CRS)
    train["x_5070"], train["y_5070"] = pts.x.to_numpy(), pts.y.to_numpy()

    # ---- report -----------------------------------------------------------
    print(f"\n=== Re-rank training table ===")
    print(f"Total rows        : {len(train)}")
    print(f"Label=1 (true)    : {int((train['label'] == 1).sum())}")
    print(f"Label=0 (competitor): {int((train['label'] == 0).sum())}")
    print(f"Distinct plants   : {train['CWNS_ID'].nunique()}")
    print(f"Pools with a positive: {int(train['pool_has_positive'].sum() // args.top_k)} "
          f"(approx)")

    pos, neg = train[train["label"] == 1], train[train["label"] == 0]
    if len(pos):
        print(f"\n  OD detection rate -- true      : "
              f"{pos['od_has_detection'].mean():.1%} (n={len(pos)})")
        print(f"  OD detection rate -- competitors: "
              f"{neg['od_has_detection'].mean():.1%} (n={len(neg)})")
        print(f"  Stage 2a rank of the true parcel: "
              f"median {pos['stage2a_rank'].median():.0f}, "
              f"top-5 {int((pos['stage2a_rank'] <= 5).sum())}/{len(pos)}")
        print(f"\n  Those two rates are what a re-ranker has to work with. If "
              f"they are close,\n  no model will separate the true parcel from "
              f"its competitors.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    C.FEATURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUT_PATH, index=False)
    print(f"\nWritten: {OUT_PATH}")
    print("\nNext: 07b_train_rerank.py")


if __name__ == "__main__":
    main()
