"""
05b_rerank_candidates.py
========================
Applies the Stage 2a re-ranker to stage2_candidates.parquet, writing a
rerank_score alongside the existing stage2_prob_correct so the two orderings
can be compared directly in review.

WHAT THIS IS FOR
    Stage 2a ranks candidates on parcel attributes alone -- it cannot run
    object detection on 4.3M candidates. The re-ranker operates on the
    surviving top-K, where OD is affordable and where, measured on four
    paired seeds (07b, 2026-09-08), it is worth about +6.5pp recall@1 over
    the same model without OD features:

        recall@1   Stage 2a 59.4%  ->  re-rank no-OD 67.6%  ->  re-rank 74.2%
        (means across seeds 42/7/13/99, same splits both arms)

    Roughly 60% of the gain needs no imagery at all. --no-od loads the
    ablation model and skips the OD join entirely, which is the deployable-
    today path when 01e coverage is incomplete.

FEATURE PARITY WITH 06b IS THE WHOLE GAME
    The model was fit on columns 06b_build_rerank_training.py constructed:
    stage2a_rank, od_pool_n_fired, od_any_fired_in_pool,
    od_fired_share_of_pool, plus the same OD fillna contract. Those are
    derived here by the same arithmetic, in the same order. A silent
    mismatch is the failure mode that matters -- score() reindexes to the
    bundle's feature_columns and raises on any column missing outright,
    but a column present with DIFFERENT semantics would score fine and be
    wrong. Keep the two definitions in sync.

    true_ll_uuid and pool_has_positive do not exist at inference and were
    dropped from training for that reason; nothing here reconstructs them.

FALLBACK
    When no candidate in a plant's pool fired, every OD column is constant
    within that competition and carries no ranking information. The model
    has od_any_fired_in_pool and can learn to lean on stage2_prob_correct
    in that case -- but a hard fallback is kept as a backstop, and plants
    resolved that way are flagged with rerank_fallback so a reviewer can
    see which ordering they are looking at. Disable with --no-fallback.

Output: data/inference/stage2_candidates_reranked.parquet
        data/inference/plant_summary_reranked.parquet

Usage:
    python 05b_rerank_candidates.py
    python 05b_rerank_candidates.py --no-od
    python 05b_rerank_candidates.py --states OH,WI,KS
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

OD_ROOT = C.DATA_DIR / "od_features_candidates" / "candidates"
INFER_DIR = C.DATA_DIR / "inference"


def load_model(no_od: bool):
    tag = "_noOD" if no_od else ""
    path = C.MODELS_DIR / f"rerank_rf_model{tag}.joblib"
    if not path.exists():
        print(f"ERROR: {path} not found. Train it with 07b_train_rerank.py"
              f"{' --no-od' if no_od else ''}.")
        sys.exit(2)
    bundle = joblib.load(path)
    print(f"  Loaded {path.name}: {len(bundle['feature_columns'])} feature columns")
    return bundle


def load_candidate_od(states: list[str] | None) -> pd.DataFrame | None:
    """01e --all output. Reads the SAME root the holdout run used -- both
    were produced by the deployed best.pt on the same parcels, so they are
    interchangeable. The _train root is deliberately NOT read: those rows
    belong to plants the model trained on."""
    if not OD_ROOT.exists() or not any(OD_ROOT.rglob("*.parquet")):
        return None
    files = sorted(OD_ROOT.rglob("*.parquet"), key=lambda f: f.stat().st_mtime)
    if states:
        want = {f"state={s}" for s in states}
        files = [f for f in files if f.parent.name in want]
    if not files:
        return None
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        for col in df.select_dtypes(include=["category"]).columns:
            df[col] = df[col].astype(str)
        frames.append(df)
    od = pd.concat(frames, ignore_index=True)
    od["CWNS_ID"] = od["CWNS_ID"].astype(str)
    od["ll_uuid"] = od["ll_uuid"].astype(str)
    od = od.drop_duplicates(subset=["CWNS_ID", "ll_uuid"], keep="last")
    return od.drop(columns=["state", "orig_lon", "orig_lat", "n_objects_total",
                            "n_objects_in_parcel", "processed_at"], errors="ignore")


def add_derived(df: pd.DataFrame, has_od: bool) -> pd.DataFrame:
    """Recreate 06b's derived columns. Order and arithmetic must match."""
    df = df.sort_values(["CWNS_ID", "stage2_prob_correct"], ascending=[True, False])
    df["stage2a_rank"] = df.groupby("CWNS_ID").cumcount() + 1

    if not has_od:
        return df

    if "od_ran" in df.columns:
        df["od_ran"] = df["od_ran"].fillna(False)
    if "od_has_detection" in df.columns:
        df["od_has_detection"] = df["od_has_detection"].fillna(False)
    for c in df.columns:
        if c.startswith("od_has_"):
            df[c] = df[c].fillna(False)
        elif c.startswith("od_n_"):
            df[c] = df[c].fillna(0)

    fired = df.groupby("CWNS_ID")["od_has_detection"].transform("sum")
    df["od_pool_n_fired"] = fired.astype(int)
    df["od_any_fired_in_pool"] = fired > 0
    fired_arr = fired.to_numpy(dtype=float)
    detected_arr = df["od_has_detection"].astype(float).to_numpy()
    share = np.zeros(len(df), dtype=float)
    np.divide(detected_arr, fired_arr, out=share, where=(fired_arr > 0))
    df["od_fired_share_of_pool"] = share
    return df


def add_projected_coords(df: pd.DataFrame) -> pd.DataFrame:
    if {"centroid_lng", "centroid_lat"}.issubset(df.columns):
        lon, lat = df["centroid_lng"], df["centroid_lat"]
    else:
        lon, lat = df["LONGITUDE"], df["LATITUDE"]
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat),
                        crs=C.EXPORT_CRS).to_crs(C.PROJECTED_CRS)
    df = df.copy()
    df["x_5070"], df["y_5070"] = pts.x.to_numpy(), pts.y.to_numpy()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-od", action="store_true",
                    help="use the ablation model and skip the OD join. The "
                         "deployable-today path when 01e --all has not "
                         "finished -- worth about 60%% of the full gain.")
    ap.add_argument("--top-k", type=int, default=20,
                    help="must match what 06b/07b trained on")
    ap.add_argument("--states", default=None, help="comma-separated filter")
    ap.add_argument("--no-fallback", action="store_true",
                    help="do not fall back to Stage 2a ordering for pools "
                         "where nothing fired")
    args = ap.parse_args()

    states = [s.strip() for s in args.states.split(",")] if args.states else None

    print("=== 05b_rerank_candidates.py ===")
    if args.no_od:
        print("*** --no-od: ablation model, no OD features ***")

    print("\nLoading re-ranker...")
    bundle = load_model(args.no_od)

    cand_path = INFER_DIR / "stage2_candidates.parquet"
    if not cand_path.exists():
        print(f"ERROR: {cand_path} not found. Run 05 (+ merge) first.")
        sys.exit(2)

    print("\nLoading Stage 2a candidates...")
    df = pd.read_parquet(cand_path)
    df["CWNS_ID"] = df["CWNS_ID"].astype(str)
    df["ll_uuid"] = df["ll_uuid"].astype(str)
    if states:
        df = df[df["STATE_CODE"].isin(states)]
        print(f"  Restricted to {states}")
    df = df.sort_values(["CWNS_ID", "stage2_prob_correct"], ascending=[True, False])
    df = df.groupby("CWNS_ID", group_keys=False).head(args.top_k)
    print(f"  {len(df)} candidate(s) across {df['CWNS_ID'].nunique()} plant(s)")

    has_od = False
    if not args.no_od:
        print("\nLoading candidate OD features (01e --all)...")
        od = load_candidate_od(states)
        if od is None:
            print(f"ERROR: no OD output under {OD_ROOT}.\n"
                  f"  Either run the 01e array, or use --no-od to score with "
                  f"the ablation model.")
            sys.exit(2)
        print(f"  {len(od)} candidate(s) with OD "
              f"across {od['CWNS_ID'].nunique()} plant(s)")
        before = len(df)
        df = df.merge(od, on=["CWNS_ID", "ll_uuid"], how="left")
        assert len(df) == before, "OD join changed row count -- duplicate keys"
        covered = df["od_ran"].notna().sum() if "od_ran" in df.columns else 0
        print(f"  OD coverage: {covered}/{len(df)} candidate(s) "
              f"({covered / len(df):.1%})")
        if covered < len(df) * 0.5:
            print("  WARNING: under half of candidates have OD. Rows without "
                  "it score as 'nothing detected', which is indistinguishable "
                  "from a genuine non-detection. Consider --no-od until the "
                  "01e array has covered these states.")
        has_od = True

    print("\nDeriving features...")
    df = add_derived(df, has_od)
    df = add_projected_coords(df)

    X = df.reindex(columns=bundle["feature_columns"])
    truly_missing = [c for c in bundle["feature_columns"] if c not in df.columns]
    if truly_missing:
        print(f"ERROR: the model expects columns absent from this frame:\n"
              f"  {truly_missing}\n"
              f"Feature engineering has drifted from 06b. Do not score around "
              f"this -- reconcile add_derived() against 06b's construction.")
        sys.exit(2)
    if "place_match" in X.columns:
        X["place_match"] = X["place_match"].fillna(False)

    print("Scoring...")
    df["rerank_score"] = bundle["pipeline"].predict_proba(X)[:, 1]

    # ---- ordering, with the fallback ------------------------------------
    df["rerank_fallback"] = False
    if has_od and not args.no_fallback:
        no_fire = df["od_pool_n_fired"] == 0
        n_plants_fb = df.loc[no_fire, "CWNS_ID"].nunique()
        df.loc[no_fire, "rerank_fallback"] = True
        # Order by Stage 2a for those pools while KEEPING rerank_score, so a
        # reviewer can still see what the model thought.
        df["_order"] = np.where(no_fire, df["stage2_prob_correct"], df["rerank_score"])
        print(f"\n  Fallback to Stage 2a ordering for {n_plants_fb} plant(s) "
              f"where no candidate fired")
    else:
        df["_order"] = df["rerank_score"]

    df = df.sort_values(["CWNS_ID", "_order"], ascending=[True, False])
    df["rerank_rank"] = df.groupby("CWNS_ID").cumcount() + 1
    df = df.drop(columns=["_order"])

    # ---- how much did the order actually change? -------------------------
    moved = (df["rerank_rank"] != df["stage2a_rank"])
    top1_changed = df[(df["rerank_rank"] == 1) & (df["stage2a_rank"] != 1)]
    print(f"\n  Candidates whose rank changed : {int(moved.sum())} "
          f"({moved.mean():.1%})")
    print(f"  Plants with a NEW top pick    : {top1_changed['CWNS_ID'].nunique()} "
          f"({top1_changed['CWNS_ID'].nunique() / df['CWNS_ID'].nunique():.1%})")
    if len(top1_changed):
        print(f"  Median Stage 2a rank of the promoted candidate: "
              f"{top1_changed['stage2a_rank'].median():.0f}")

    # ---- outputs ---------------------------------------------------------
    tag = "_noOD" if args.no_od else ""
    out_c = INFER_DIR / f"stage2_candidates_reranked{tag}.parquet"
    df.to_parquet(out_c, index=False)
    print(f"\nWritten: {out_c}")

    top = df[df["rerank_rank"] == 1][
        ["CWNS_ID", "ll_uuid", "rerank_score", "stage2_prob_correct",
         "stage2a_rank", "rerank_fallback"]].rename(columns={
            "ll_uuid": "rerank_top_ll_uuid",
            "stage2a_rank": "rerank_top_stage2a_rank"})

    summary_path = INFER_DIR / "plant_summary.parquet"
    if summary_path.exists():
        summ = pd.read_parquet(summary_path)
        summ["CWNS_ID"] = summ["CWNS_ID"].astype(str)
        summ = summ.merge(top.drop(columns=["stage2_prob_correct"]),
                          on="CWNS_ID", how="left")
        out_s = INFER_DIR / f"plant_summary_reranked{tag}.parquet"
        summ.to_parquet(out_s, index=False)
        print(f"Written: {out_s}")

    print("\nNext: point 10_build_review_queue.py at the reranked table. Both "
          "scores are\npreserved per candidate, so a reviewer can compare "
          "Stage 2a against the re-rank.")


if __name__ == "__main__":
    main()
