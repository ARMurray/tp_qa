"""
calibrate_move_rule.py
======================
Out-of-sample precision of the automated move, by re-ranker cutoff.

THE RULE BEING CALIBRATED
    Move a plant to the re-ranker's #1 parcel when Stage 1 flagged its
    reported location (it is in the candidate_pick task) AND that #1's
    re-rank score >= cutoff. The project target is > 90% of moves right.

WHAT COUNTS
    Only review rounds the models had not trained on when the queue was built
    (round 4 onward -- earlier rounds' plants were folded into training, so
    scoring them is in-sample). One row per decided candidate_pick plant:

      right             reviewer picked the #1 candidate
      wrong_candidate   reviewer picked another candidate (location was wrong,
                        the move goes to the wrong parcel)
      truth_outside     the truth is in no candidate (location was wrong, so is
                        the move)
      false_move        the reported location was RIGHT -- the worst error:
                        a correct record would be overwritten
      needs_info        undecided; excluded from precision, counted separately

    Precision = right / (right + the three wrong kinds), with a Wilson 95% CI.
    A cutoff meets the target only when the CI's LOWER bound clears it.

CAVEATS
    - Queues oversample the 'uncertain' slice (small margins), so a cutoff's
      precision is conditional on reaching review, not a national estimate.
      The per-slice split is printed for that reason.
    - Each round was scored by the models of its own day. Pooling rounds
      pools model versions; the per-round tables show whether they agree.

Runs anywhere with pandas -- the inputs are the committed review logs.
Usage:
    python calibrate_move_rule.py                 # rounds 4+
    python calibrate_move_rule.py --rounds 4 5
    python calibrate_move_rule.py --target 0.9 --out ../diagnostics/move_rule_calibration.txt
"""
import argparse
import io
import math
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / "review_app" / "data" / "outgoing"
FIRST_OUT_OF_SAMPLE_ROUND = 4
CUTOFFS = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97]
WRONG = ["wrong_candidate", "truth_outside", "false_move"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def available_rounds() -> list[int]:
    out = []
    for f in LOG_DIR.glob("review_log_round*.parquet"):
        m = re.fullmatch(r"review_log_round(\d+)\.parquet", f.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def load_round(n: int) -> pd.DataFrame:
    plants = pd.read_parquet(LOG_DIR / f"review_log_round{n}.parquet")
    cands = pd.read_parquet(LOG_DIR / f"review_log_candidates_round{n}.parquet")
    pick = plants[plants["review_task"] == "candidate_pick"].copy()
    top = cands[cands["candidate_rank"] == 1][
        ["cwns_id", "stage2b_score", "stage2a_score", "rerank_fallback",
         "od_has_detection"]]
    df = pick.merge(top, on="cwns_id", how="left")

    v = df["plant_verdict"]
    df["outcome"] = "needs_info"
    df.loc[v == "reported_correct", "outcome"] = "false_move"
    df.loc[v == "truth_outside_candidates", "outcome"] = "truth_outside"
    df.loc[v == "candidate_correct", "outcome"] = "wrong_candidate"
    df.loc[(v == "candidate_correct") & (df["candidate_rank"] == 1), "outcome"] = "right"
    df["round"] = n

    # A round whose 05b ran without the queue's detections (fallback flagged
    # while a shown candidate fired -- round 5, 2026-10-04) carries Stage 2a
    # order and OD-blind re-rank scores. Its verdicts are fine; its scores
    # are not the production re-ranker's and must be recomputed.
    fired = cands.groupby("cwns_id")["od_has_detection"].max().fillna(0).astype(bool)
    fb = cands.groupby("cwns_id")["rerank_fallback"].max().fillna(0).astype(bool)
    bad = int((fired & fb).sum())
    if bad:
        print(f"WARNING: round {n}: {bad} plant(s) marked 'nothing fired' with a "
              f"detection attached -- its re-rank scores were computed without "
              f"the queue's detections. Do NOT use its precision table until it "
              f"is re-scored with the national 05b output.\n")
    return df


def load_round_rescored(n: int, output: pd.DataFrame) -> pd.DataFrame:
    """Judge round n's verdicts against the PRODUCTION re-rank instead of the
    scores the queue carried (13's output: rerank_top_ll_uuid, rerank_score
    per plant). Needed when the queue's own re-rank was invalid (round 5) or
    the models have changed since the queue was built -- provided the round
    was NOT trained on (round 5 vs the 2026-10-03 models: not trained on).

    Every decided plant the production pipeline would act on counts --
    candidate_pick AND confirm_reported -- because the current Stage 1 may
    flag a plant the queue-time model passed:
      reported_correct          any move is a false_move
      candidate_correct         right iff production #1 == the reviewer's parcel
      truth_outside_candidates  counted WRONG (truth_outside). Conservative:
                                the truth lay outside the 5 shown, but the
                                production #1 could still be the truth parcel;
                                no parcel geometry here to check.
    """
    plants = pd.read_parquet(LOG_DIR / f"review_log_round{n}.parquet")
    if "reviewed" in plants.columns:
        plants = plants[plants["reviewed"] == 1]
    out = output[["CWNS_ID", "rerank_top_ll_uuid", "rerank_score", "rerank_fallback"]] \
        .rename(columns={"CWNS_ID": "cwns_id", "rerank_score": "stage2b_score"})
    df = plants.merge(out, on="cwns_id", how="inner")
    df = df[df["stage2b_score"].notna()].copy()   # flagged with candidates today

    v = df["plant_verdict"]
    df["outcome"] = "needs_info"
    df.loc[v == "reported_correct", "outcome"] = "false_move"
    df.loc[v == "truth_outside_candidates", "outcome"] = "truth_outside"
    pick = v == "candidate_correct"
    same = df["selected_ll_uuid"].astype(str) == df["rerank_top_ll_uuid"].astype(str)
    df.loc[pick, "outcome"] = "wrong_candidate"
    df.loc[pick & same, "outcome"] = "right"
    df["round"] = n
    print(f"round {n} RE-SCORED against the production re-rank: "
          f"{len(df)} reviewed plant(s) flagged today with candidates "
          f"(of {len(plants)} reviewed)")
    return df


def table(df: pd.DataFrame, target: float) -> None:
    print(f"  {'cutoff':>6} {'moves':>6} {'right':>6} {'prec':>7} "
          f"{'95% CI':>13} {'wrongC':>6} {'outside':>7} {'FALSE':>6} {'n_info':>6}")
    for t in CUTOFFS:
        s = df[df["stage2b_score"] >= t]
        c = s["outcome"].value_counts()
        k = int(c.get("right", 0))
        n = k + sum(int(c.get(w, 0)) for w in WRONG)
        lo, hi = wilson(k, n)
        flag = "  <-- CI clears target" if n and lo >= target else ""
        prec = f"{k / n:6.1%}" if n else "    --"
        print(f"  {t:>6.2f} {n:>6} {k:>6} {prec:>7} "
              f"{lo:>5.0%}-{hi:<5.0%}  {int(c.get('wrong_candidate', 0)):>6} "
              f"{int(c.get('truth_outside', 0)):>7} {int(c.get('false_move', 0)):>6} "
              f"{int(c.get('needs_info', 0)):>6}{flag}")


def plants_needed(p: float, target: float) -> int | None:
    """Smallest n at which observed precision p would put Wilson's lower
    bound at or above target."""
    if p <= target:
        return None
    for n in range(5, 5000):
        if wilson(round(p * n), n)[0] >= target:
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="*", default=None)
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--rescore", type=int, nargs="*", default=[],
                    help="rounds to judge against the production re-rank "
                         "(13's output) instead of the queue's own scores")
    ap.add_argument("--output", type=Path,
                    default=REPO / "correction" / "diagnostics" / "output"
                    / "cwns_corrected_locations.parquet",
                    help="13's output, for --rescore")
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the report to this file")
    args = ap.parse_args()

    rounds = args.rounds or [r for r in available_rounds()
                             if r >= FIRST_OUT_OF_SAMPLE_ROUND]
    missing = [r for r in rounds
               if not (LOG_DIR / f"review_log_candidates_round{r}.parquet").exists()]
    if missing:
        print(f"ERROR: no review logs for round(s) {missing} in {LOG_DIR}")
        sys.exit(2)

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("=== calibrate_move_rule.py ===")
        print(f"rounds: {rounds}  |  target precision: {args.target:.0%}  "
              f"(met only when the 95% CI lower bound clears it)\n")
        output = None
        if args.rescore:
            output = pd.read_parquet(args.output)
            output["CWNS_ID"] = output["CWNS_ID"].astype(str)
            print(f"production re-rank from {args.output.name} "
                  f"(built {output['built'].iloc[0] if 'built' in output else '?'})\n")
        frames = [load_round_rescored(r, output) if r in args.rescore else load_round(r)
                  for r in rounds]
        all_ = pd.concat(frames, ignore_index=True)

        for r, df in zip(rounds, frames):
            if r in args.rescore:
                print(f"--- round {r}  (RE-SCORED: production re-rank, "
                      f"{len(df)} plants flagged today) ---")
            else:
                mv = df["model_version"].dropna().unique()
                print(f"--- round {r}  (queue-time model {', '.join(map(str, mv)) or '?'}; "
                      f"{len(df)} candidate_pick plants) ---")
            table(df, args.target)
            print()

        if len(rounds) > 1:
            print(f"--- pooled, rounds {rounds} ({len(all_)} plants) ---")
            table(all_, args.target)
            print()

        print("--- by queue slice, pooled ---")
        for sl, df in all_.groupby("queue_slice"):
            print(f"  [{sl}]  {len(df)} plants")
            table(df, args.target)
        print()

        print("--- re-rank fallback (#1 is Stage 2a's, nothing fired) ---")
        for fb, df in all_.groupby(all_["rerank_fallback"].fillna(0).astype(int)):
            print(f"  [rerank_fallback={fb}]  {len(df)} plants")
            table(df, args.target)
        print()

        print("--- plants needed for the CI to clear the target ---")
        for p in (0.93, 0.95, 0.97, 0.99):
            n = plants_needed(p, args.target)
            print(f"  observed precision {p:.0%}: n >= {n} moves at the cutoff")

    text = buf.getvalue()
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
