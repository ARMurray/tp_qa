"""
check_review_queue_scores.py
=============================
Verifies review_queue_round{N}.parquet actually carries both stage2a_score
and stage2b_score (plus rerank_fallback) for candidate_pick rows, after the
10_build_review_queue.py changes to read from the reranked inference output.

Usage:
    python check_review_queue_scores.py --round 2
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()

    path = C.DATA_DIR / "review_queue" / f"review_queue_round{args.round}.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)

    df = pd.read_parquet(path)
    print(f"Loaded {path}")
    print(f"Columns: {df.columns.tolist()}\n")

    has_2b = "stage2b_score" in df.columns
    has_fb = "rerank_fallback" in df.columns
    print(f"stage2b_score present  : {has_2b}")
    print(f"rerank_fallback present: {has_fb}\n")

    if not has_2b:
        print("STOP: stage2b_score is missing entirely -- the 10_ edits didn't "
              "take, or this file predates them. Nothing further to check.")
        sys.exit(1)

    cp = df[df["review_task"] == "candidate_pick"]
    print(f"candidate_pick rows: {len(cp)}")
    print(f"  stage2a_score null: {cp['stage2a_score'].isna().sum()} / {len(cp)}")
    print(f"  stage2b_score null: {cp['stage2b_score'].isna().sum()} / {len(cp)}")
    if has_fb:
        print(f"  rerank_fallback True: {cp['rerank_fallback'].sum()} / {len(cp)}")

    print("\nSample rows:")
    cols = ["CWNS_ID", "candidate_rank", "stage2a_score", "stage2b_score",
            "score_margin"] + (["rerank_fallback"] if has_fb else [])
    print(cp[cols].head(10).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
