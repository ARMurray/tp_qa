"""
export_viewer_data.py
=====================
Slim copies of the scored candidates and every detected object, for the
viewer (viewer/) on the work computer. Run by 13_build_corrected_output.slurm
right after 13, so each cycle refreshes them; can also be run on its own.

Writes to correction/diagnostics/output/ (committed, overwritten each run):

  viewer_candidates.parquet   one row per (plant, candidate parcel) that
                              05/05b scored -- ALL of them, not only #1 or
                              those above the move cutoff. Ranks, Stage 2a and
                              re-rank scores, distance, the detection summary,
                              and the parcel facts the models used.
  viewer_objects.parquet      one row per detected object from the deployed
                              detector: around reported locations (01b),
                              corrected locations (01c) and candidate parcels
                              (01e). lon/lat, class, confidence, source, and
                              the parcel it was run on when there is one.

Parcel GEOMETRY is deliberately not exported: the viewer reads it live from
the local Regrid mirror. Float columns are float32 and the files are zstd
compressed: ~189k candidate rows come to a few MB.

    python export_viewer_data.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

INFER_DIR = C.DATA_DIR / "inference"
SHARE_DIR = C.ROOT / "diagnostics" / "output"

CAND_COLS = [
    "CWNS_ID", "STATE_CODE", "ll_uuid",
    "rerank_rank", "rerank_score", "rerank_fallback",
    "stage2a_rank", "stage2_prob_correct",
    "distance_m", "centroid_lat", "centroid_lng",
    "od_ran", "od_has_detection", "od_n_objects", "od_max_confidence", "od_dominant_class",
    "name_match_score", "has_ww_keyword", "osm_ww", "utility_owner",
    "ll_gisacre", "ll_bldg_count", "dominant_class_group",
]

OBJECT_ROOTS = {
    "reported": C.OD_OUTPUT_DIR / "objects",
    "corrected": C.OD_OUTPUT_DIR_CORRECTED / "objects",
    "candidate": C.DATA_DIR / "od_features_candidates" / "objects",
    "candidate_queue": C.DATA_DIR / "od_features_candidates_queue" / "objects",
}
OBJ_COLS = ["CWNS_ID", "ll_uuid", "lon", "lat", "class_name", "max_confidence"]


def slim(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 (coordinates keep ~1 m at float32 only near zero, so
    lon/lat columns stay float64)."""
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]) and c not in ("lon", "lat", "centroid_lat", "centroid_lng"):
            df[c] = df[c].astype("float32")
    return df


def deployed_model_mtime() -> float | None:
    pts = sorted(C.OD_MODEL_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime,
                 reverse=True) if C.OD_MODEL_DIR.exists() else []
    return pts[0].stat().st_mtime if pts else None


def export_candidates() -> None:
    src = INFER_DIR / "stage2_candidates_reranked.parquet"
    if not src.exists():
        src = INFER_DIR / "stage2_candidates.parquet"
        print(f"  NOTE: no re-ranked candidates; exporting Stage 2a only from {src.name}")
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(src).schema_arrow.names)
    cols = [c for c in CAND_COLS if c in have]
    missing = [c for c in CAND_COLS if c not in have]
    df = pd.read_parquet(src, columns=cols)
    df["CWNS_ID"] = df["CWNS_ID"].astype(str)
    df["ll_uuid"] = df["ll_uuid"].astype(str)
    if "stage2a_rank" not in df.columns and "stage2_prob_correct" in df.columns:
        df["stage2a_rank"] = df.groupby("CWNS_ID")["stage2_prob_correct"] \
                               .rank(ascending=False, method="first")
    for c in ("rerank_rank", "stage2a_rank", "od_n_objects", "ll_bldg_count"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int32")
    df = slim(df)
    out = SHARE_DIR / "viewer_candidates.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    print(f"  viewer_candidates.parquet: {len(df):,} rows, {df['CWNS_ID'].nunique():,} plants, "
          f"{out.stat().st_size / 1e6:.1f} MB  (from {src.name})")
    if missing:
        print(f"    columns not in the source, skipped: {missing}")


def export_objects() -> None:
    model_t = deployed_model_mtime()
    frames = []
    for source, root in OBJECT_ROOTS.items():
        if not root.exists():
            continue
        files = [f for f in root.rglob("*.parquet")
                 if model_t is None or f.stat().st_mtime >= model_t]
        n = 0
        for f in files:
            try:
                import pyarrow.parquet as pq
                have = set(pq.ParquetFile(f).schema_arrow.names)
                d = pd.read_parquet(f, columns=[c for c in OBJ_COLS if c in have])
            except Exception:
                continue
            if "ll_uuid" not in d.columns:
                d["ll_uuid"] = None
            d["source"] = source
            frames.append(d)
            n += len(d)
        print(f"  objects from {source:<16} {n:>9,}  ({len(files)} current file(s))")
    if not frames:
        print("  no detected objects found")
        return
    df = pd.concat(frames, ignore_index=True)
    df["CWNS_ID"] = df["CWNS_ID"].astype(str)
    df["ll_uuid"] = df["ll_uuid"].astype("string")
    # The same object can be written twice (a candidate run from the national
    # and the queue root, or a resumed run): keep one per plant/parcel/spot/class.
    df["_k"] = (df["lon"].round(5).astype(str) + df["lat"].round(5).astype(str))
    df = df.drop_duplicates(subset=["CWNS_ID", "ll_uuid", "_k", "class_name"]).drop(columns="_k")
    df = df.rename(columns={"max_confidence": "confidence"})
    df["source"] = df["source"].astype("category")
    df["class_name"] = df["class_name"].astype("category")
    df = slim(df)
    out = SHARE_DIR / "viewer_objects.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    print(f"  viewer_objects.parquet: {len(df):,} objects, {out.stat().st_size / 1e6:.1f} MB")


def main():
    print("=== export_viewer_data.py ===")
    SHARE_DIR.mkdir(parents=True, exist_ok=True)
    export_candidates()
    export_objects()
    print(f"Written to {SHARE_DIR} -- commit correction/diagnostics/output/ and "
          f"git pull on the work computer; the viewer picks them up on restart.")


if __name__ == "__main__":
    main()
