"""
10_build_review_queue.py
==========================
Assembles a reviewable batch from 05_run_inference.py's output, per
REVIEW_LOOP_PLAN.md Phase 3.

DEVIATION FROM THE PLAN, INTENTIONAL: the plan's queue schema is built around
Stage 2b scores (`stage2b_score`, margin between Stage 2b's top-1/top-2).
Stage 2b is deliberately excluded from this pilot round -- see
TPQA_MASTER_REFERENCE.md S3/S10: OD features contribute almost nothing to the
current Stage 2b model, so it isn't in 05's inference chain yet. Every place
the plan says "Stage 2b score" this script uses Stage 2a's `stage2_prob_correct`
instead, and says so in the column names (`stage2a_score`, not `stage2b_score`)
so nothing pretends otherwise. If Stage 2b re-enters the pipeline later, this
script's scoring basis needs to change, not just its labels.

TWO REVIEW TASK TYPES, not one -- the plan's per-candidate schema assumes
every queued plant has candidates to choose from. It doesn't, for a reason
the plan itself calls out: the "random" slice exists specifically to catch
CONFIDENT-AND-WRONG STAGE 1 FAILURES, and a plant Stage 1 passed as correct
never went through Stage 2a and has no candidates at all. So:
  - candidate_pick : flagged plants (trigger_reason != "none") with >=1
                      candidate. Reviewer picks from top-K, or says the truth
                      isn't in the list.
  - confirm_reported: plants Stage 1 passed (trigger_reason == "none"), OR
                      flagged plants that lost every candidate (K_RINGS +
                      reported-parcel exclusion left nothing). Reviewer
                      confirms or rejects the REPORTED location directly --
                      there is nothing to rank.
Both task types can appear in the "random" slice. Only candidate_pick can
appear in "uncertain" (there's no uncertainty to measure without candidates).
"holdout" draws from whichever type each holdout plant naturally falls into.

QUEUE COMPOSITION per batch (Phase 3 table):
  holdout    : round 1 only, unlabeled-bin manifest members needing their
               one-time review, not counted against the uncertain/random split
  uncertain  : ~70% of the remainder -- small top1/top2 margin, or Stage 1
               says wrong but best candidate score is low, or nothing above
               the Stage 2a threshold
  random     : ~30% of the remainder -- uniform over ALL scored plants
               (both task types), not just flagged ones. Not droppable.

Usage:
    python 10_build_review_queue.py --states OH,MS,DE --round 1
    python 10_build_review_queue.py --states OH,MS,DE --round 1 --batch-size 150
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

TOP_K_SHOWN = 5           # candidates actually shown to the reviewer per plant
UNCERTAIN_SHARE = 0.70
RANDOM_SHARE = 0.30
MARGIN_UNCERTAIN_THRESHOLD = 0.10   # top1 - top2 score below this = uncertain
LOW_SCORE_UNCERTAIN_THRESHOLD = 0.60  # best candidate below this = uncertain


def load_stage2a_threshold() -> float:
    path = C.MODELS_DIR / "stage2_optimal_threshold.joblib"
    if not path.exists():
        print(f"  WARNING: {path} not found -- 'no candidate above threshold' "
              f"uncertainty rule will be skipped")
        return None
    return float(joblib.load(path))


def load_holdout_unreviewed() -> set:
    """CWNS_IDs of unlabeled-bin holdout members that don't yet have truth --
    the ones round 1 needs reviewed to make holdout scoring possible at all.
    Labeled-bin holdout members (correct/corrections) already have derivable
    truth and don't need review; see 09_build_holdout.py."""
    manifest_path = C.DATA_DIR / "holdout" / "holdout_manifest.parquet"
    truth_path = C.DATA_DIR / "holdout" / "holdout_truth.parquet"
    if not manifest_path.exists():
        print(f"  WARNING: {manifest_path} not found -- no holdout slice this round")
        return set()
    manifest = pd.read_parquet(manifest_path)
    unlabeled = set(manifest.loc[manifest["bin"] == "unlabeled", "CWNS_ID"].astype(str))
    if truth_path.exists() and len(unlabeled):
        truth = pd.read_parquet(truth_path)
        already_reviewed = set(truth["CWNS_ID"].astype(str))
        unlabeled -= already_reviewed
    return unlabeled


def build_candidate_pick_rows(plant_summary: pd.DataFrame,
                              candidates: pd.DataFrame) -> pd.DataFrame:
    """One row per (plant, candidate), top TOP_K_SHOWN by stage2a_score,
    plus candidate_rank and score_margin (top1 vs top2 for that plant)."""
    cand = candidates.copy()
    cand = cand.rename(columns={"stage2_prob_correct": "stage2a_score"})
    cand = cand.sort_values(["CWNS_ID", "stage2a_score"], ascending=[True, False])
    cand["candidate_rank"] = cand.groupby("CWNS_ID").cumcount() + 1
    cand = cand[cand["candidate_rank"] <= TOP_K_SHOWN].copy()

    margins = (cand[cand["candidate_rank"].isin([1, 2])]
               .pivot(index="CWNS_ID", columns="candidate_rank", values="stage2a_score"))
    if 2 in margins.columns:
        margins["score_margin"] = margins[1] - margins[2]
    else:
        # only one candidate for every plant in this batch -- undefined margin,
        # treated as maximally uncertain (nothing to compare against) rather
        # than as confidently separated.
        margins["score_margin"] = 0.0
    margins = margins[["score_margin"]].reset_index()

    cand = cand.merge(margins, on="CWNS_ID", how="left")
    cand["review_task"] = "candidate_pick"
    return cand


def build_confirm_reported_rows(plant_summary: pd.DataFrame,
                                cwns_ids: set) -> pd.DataFrame:
    """One row per plant with nothing to rank -- reviewer confirms/rejects the
    reported location directly. Covers both Stage-1-passed plants (no Stage 2a
    run at all) and flagged plants that lost every candidate."""
    rows = plant_summary[plant_summary["CWNS_ID"].isin(cwns_ids)].copy()
    rows["candidate_rank"] = pd.NA
    rows["ll_uuid"] = rows["reported_ll_uuid"]
    rows["stage2a_score"] = np.nan
    rows["score_margin"] = np.nan
    rows["review_task"] = "confirm_reported"
    keep_cols = ["CWNS_ID", "STATE_CODE", "ll_uuid", "candidate_rank",
                 "stage2a_score", "score_margin", "review_task"]
    return rows[[c for c in keep_cols if c in rows.columns]]


def classify_uncertain(plant_summary: pd.DataFrame, cand_pick_rows: pd.DataFrame,
                       stage2a_threshold: float | None) -> set:
    """Plants eligible for the 'uncertain' slice -- candidate_pick only, per
    the module docstring. Three OR'd rules, matching Phase 3's table with
    stage2a substituted for stage2b."""
    if cand_pick_rows.empty:
        return set()

    per_plant = cand_pick_rows.groupby("CWNS_ID").agg(
        top_score=("stage2a_score", "max"),
        margin=("score_margin", "first"),
    ).reset_index()

    small_margin = per_plant.loc[per_plant["margin"] < MARGIN_UNCERTAIN_THRESHOLD, "CWNS_ID"]
    low_top_score = per_plant.loc[per_plant["top_score"] < LOW_SCORE_UNCERTAIN_THRESHOLD, "CWNS_ID"]

    below_threshold = set()
    if stage2a_threshold is not None:
        below_threshold = set(per_plant.loc[
            per_plant["top_score"] < stage2a_threshold, "CWNS_ID"])

    return set(small_margin) | set(low_top_score) | below_threshold


def load_plant_display_info(plant_summary: pd.DataFrame) -> pd.DataFrame:
    """Descriptive info for the docked plant-info panel -- population served
    and address/city/county for identifying the plant, plus a few discharge
    facts. Two sources, both re-read directly here rather than plumbed
    through 05_run_inference.py's output, since plant_summary.parquet was
    deliberately kept narrow (see 05's docstring) and reopening that
    contract risked destabilizing an already-tested script.

    NOTE, UNVERIFIED: no CWNS table I've seen has an explicit facility NAME
    field -- only ADDRESS/CITY/COUNTY_NAME/ZIP_CODE (from PHYSICAL_LOCATION.
    txt, confirmed via CWNSDatabaseDictionaryJanuary2025.xlsx). Using those as
    the identifying info for now. If a name table exists elsewhere in the
    CWNS export and I've missed it, this should be extended, not replaced --
    address/city stay useful regardless."""
    ids = set(plant_summary["CWNS_ID"].astype(str))

    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt", dtype=str, encoding="latin1")
    loc = loc[loc["CWNS_ID"].isin(ids)]
    loc_cols = ["CWNS_ID", "ADDRESS", "CITY", "COUNTY_NAME", "ZIP_CODE"]
    loc = loc[[c for c in loc_cols if c in loc.columns]].drop_duplicates(subset="CWNS_ID")

    plant_features_path = C.FEATURES_OUTPUT_DIR / "05_plant_features.parquet"
    pf_cols = ["CWNS_ID", "pop_served", "subdivision", "place", "county", "is_rural",
               "surface_water_discharge", "requires_npdes", "any_reuse"]
    if plant_features_path.exists():
        pf = pd.read_parquet(plant_features_path)
        pf["CWNS_ID"] = pf["CWNS_ID"].astype(str)
        pf = pf[pf["CWNS_ID"].isin(ids)]
        pf = pf[[c for c in pf_cols if c in pf.columns]].drop_duplicates(subset="CWNS_ID")
    else:
        print(f"  WARNING: {plant_features_path} not found -- pop_served/discharge "
              f"info will be missing from the plant-info panel")
        pf = pd.DataFrame(columns=pf_cols)

    return loc.merge(pf, on="CWNS_ID", how="outer")


def load_previously_reviewed_ids() -> set:
    """Every CWNS_ID that appears in ANY previously-uploaded review_log_round*
    or holdout_truth_round* parquet, regardless of round -- these are the
    same files sync/push_review_log.py already writes and you already
    MobaXterm-upload for 11_ingest_review_log.py, so this needs no new
    plumbing. Covers every verdict type including needs_info: a plant that
    was reviewed and skipped as needs_info still shouldn't silently
    resurface in the very next round's random/uncertain draw -- if you want
    to deliberately re-queue it, that's a choice to make explicitly, not a
    default.

    Added 2026-08-28 after realizing this script had no awareness of prior
    rounds at all -- holdout already excluded correctly (round>1 skips that
    slice), uncertain/random did not, and nothing stopped a round 2 draw
    from re-selecting plants already reviewed in round 1."""
    incoming_dir = C.DATA_DIR / "review_log_incoming"
    if not incoming_dir.exists():
        return set()
    ids = set()
    for pattern in ("review_log_round*.parquet", "holdout_truth_round*.parquet"):
        for path in incoming_dir.glob(pattern):
            try:
                df = pd.read_parquet(path, columns=None)
            except Exception as e:
                print(f"  WARNING: could not read {path.name} for prior-round "
                      f"exclusion: {e}")
                continue
            id_col = "cwns_id" if "cwns_id" in df.columns else "CWNS_ID"
            ids |= set(df[id_col].astype(str))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=str, required=True,
                     help="must match the states 05_run_inference.py was run for")
    ap.add_argument("--round", type=int, required=True,
                     help="review round number. Round 1 includes the holdout slice; "
                          "later rounds should not re-queue holdout plants.")
    ap.add_argument("--batch-size", type=int, default=150)
    ap.add_argument("--model-version", type=str, default=None,
                     help="tag for provenance. Defaults to today's date if omitted -- "
                          "there is no formal model-versioning scheme yet "
                          "(round_manifest.json doesn't exist yet either, see "
                          "TPQA_MASTER_REFERENCE.md S4 Phase 7).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datetime import date
    model_version = args.model_version or f"unversioned-{date.today().isoformat()}"

    print("=== 10_build_review_queue.py ===")
    print(f"States: {args.states}  |  Round: {args.round}  |  "
          f"Model version tag: {model_version}")

    inference_dir = C.DATA_DIR / "inference"
    plant_summary = pd.read_parquet(inference_dir / "plant_summary.parquet")
    candidates = pd.read_parquet(inference_dir / "stage2_candidates.parquet")
    print(f"\nLoaded {len(plant_summary)} scored plants, "
          f"{len(candidates)} candidate rows")

    # Exclude anything already reviewed in a prior round BEFORE any slice
    # logic runs, so it's structurally impossible for a downstream slice to
    # re-select it -- filtering plant_summary itself rather than filtering
    # each slice separately.
    already_reviewed = load_previously_reviewed_ids()
    if already_reviewed:
        n_before = len(plant_summary)
        plant_summary = plant_summary[~plant_summary["CWNS_ID"].isin(already_reviewed)].copy()
        candidates = candidates[~candidates["CWNS_ID"].isin(already_reviewed)].copy()
        print(f"  Excluding {n_before - len(plant_summary)} plant(s) already "
              f"reviewed in a prior round ({len(already_reviewed)} total prior "
              f"reviews found across all uploaded review_log/holdout_truth files)")

    print("\nLoading plant display info (address/city/county, pop_served, "
          "discharge facts)...")
    plant_display = load_plant_display_info(plant_summary)
    plant_summary = plant_summary.merge(plant_display, on="CWNS_ID", how="left")

    stage2a_threshold = load_stage2a_threshold()

    # ---- Build both review-task row sets ----
    flagged_with_cands = set(
        plant_summary.loc[plant_summary["n_candidates"] > 0, "CWNS_ID"])
    cand_pick_rows = build_candidate_pick_rows(
        plant_summary, candidates[candidates["CWNS_ID"].isin(flagged_with_cands)])

    confirm_ids = set(plant_summary["CWNS_ID"]) - flagged_with_cands
    confirm_rows = build_confirm_reported_rows(plant_summary, confirm_ids)
    print(f"\nReview tasks: {plant_summary['CWNS_ID'].nunique()} plants -- "
          f"{len(flagged_with_cands)} candidate_pick, {len(confirm_ids)} confirm_reported")

    # ---- Holdout slice (round 1 only) ----
    holdout_ids = load_holdout_unreviewed() if args.round == 1 else set()
    holdout_ids &= set(plant_summary["CWNS_ID"])  # scope to this pilot's states
    print(f"\nHoldout slice: {len(holdout_ids)} unreviewed unlabeled-bin plant(s) "
          f"in this pilot's scope" + ("" if args.round == 1 else " (skipped -- round > 1)"))

    # ---- Uncertain / random, drawn from the NON-holdout pool ----
    non_holdout_ids = set(plant_summary["CWNS_ID"]) - holdout_ids
    uncertain_ids = classify_uncertain(
        plant_summary, cand_pick_rows[cand_pick_rows["CWNS_ID"].isin(non_holdout_ids)],
        stage2a_threshold) & non_holdout_ids

    remaining_slots = max(0, args.batch_size - len(holdout_ids))
    n_uncertain_target = round(remaining_slots * UNCERTAIN_SHARE)
    n_random_target = remaining_slots - n_uncertain_target

    rng = np.random.default_rng(args.seed)
    uncertain_pool = sorted(uncertain_ids)  # sorted for reproducibility pre-shuffle
    rng.shuffle(uncertain_pool)
    uncertain_selected = set(uncertain_pool[:n_uncertain_target])

    random_pool = sorted(non_holdout_ids - uncertain_selected)
    rng.shuffle(random_pool)
    random_selected = set(random_pool[:n_random_target])

    print(f"\nBatch composition (target size {args.batch_size}):")
    print(f"  holdout   : {len(holdout_ids)}")
    print(f"  uncertain : {len(uncertain_selected)} (of {len(uncertain_ids)} eligible, "
          f"target {n_uncertain_target})")
    print(f"  random    : {len(random_selected)} (target {n_random_target})")
    if len(uncertain_selected) < n_uncertain_target:
        shortfall = n_uncertain_target - len(uncertain_selected)
        print(f"  NOTE: uncertain pool undersized by {shortfall} -- consider widening "
              f"MARGIN_UNCERTAIN_THRESHOLD/LOW_SCORE_UNCERTAIN_THRESHOLD, or accept a "
              f"smaller batch. Not silently backfilling from random -- that would quietly "
              f"change what 'uncertain' means batch to batch.")

    queue_plant_ids = holdout_ids | uncertain_selected | random_selected
    slice_map = {**{i: "holdout" for i in holdout_ids},
                 **{i: "uncertain" for i in uncertain_selected},
                 **{i: "random" for i in random_selected}}

    # ---- Assemble final queue: candidate_pick rows + confirm_reported rows,
    #      for exactly the selected plants, tagged with queue_slice/provenance ----
    cp = cand_pick_rows[cand_pick_rows["CWNS_ID"].isin(queue_plant_ids)].copy()
    cr = confirm_rows[confirm_rows["CWNS_ID"].isin(queue_plant_ids)].copy()

    # PARCEL_INFO_COLS: was silently dropped by an earlier version of this
    # script (common_cols only kept CWNS_ID/STATE_CODE/ll_uuid/candidate_rank/
    # stage2a_score/score_margin/review_task) even though owner/lbcs_*/acreage
    # ARE present in cand_pick_rows the whole time -- they're merged in by
    # 05_run_inference.py's run_stage2a() and never removed until this
    # script's final column select threw them away. Confirmed 2026-08-26 by a
    # live reviewer asking for exactly this info and it not being there.
    PARCEL_INFO_COLS = ["owner", "lbcs_activity", "lbcs_ownership", "lbcs_function",
                        "lbcs_structure", "lbcs_site", "ll_gisacre", "ll_bldg_count",
                        "dominant_class_group", "has_ww_keyword", "osm_ww",
                        "distance_m", "within_1km", "within_5km", "data_quality_score"]
    common_cols = ["CWNS_ID", "STATE_CODE", "ll_uuid", "candidate_rank",
                   "stage2a_score", "score_margin", "review_task"]
    all_row_cols = common_cols + PARCEL_INFO_COLS
    for df in (cp, cr):
        for c in all_row_cols:
            if c not in df.columns:
                df[c] = pd.NA

    queue = pd.concat([cp[all_row_cols], cr[all_row_cols]], ignore_index=True)
    queue["queue_slice"] = queue["CWNS_ID"].map(slice_map)
    queue["model_version"] = model_version
    queue["review_round"] = args.round
    queue["is_holdout"] = queue["queue_slice"] == "holdout"

    # PLANT_INFO_COLS: identifying/descriptive info for the docked plant-info
    # panel. See load_plant_display_info()'s docstring for the "no facility
    # NAME field found" caveat.
    PLANT_INFO_COLS = ["ADDRESS", "CITY", "COUNTY_NAME", "ZIP_CODE",
                       "pop_served", "subdivision", "place", "county", "is_rural",
                       "surface_water_discharge", "requires_npdes", "any_reuse"]
    plant_cols_present = [c for c in PLANT_INFO_COLS if c in plant_summary.columns]

    queue = queue.merge(
        plant_summary[["CWNS_ID", "LATITUDE", "LONGITUDE", "reported_ll_uuid",
                       "stage1_prob_correct", "trigger_reason"] + plant_cols_present],
        on="CWNS_ID", how="left")

    queue = queue.sort_values(
        ["queue_slice", "CWNS_ID", "candidate_rank"],
        na_position="first"
    ).reset_index(drop=True)

    out_dir = C.DATA_DIR / "review_queue"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"review_queue_round{args.round}.parquet"
    queue.to_parquet(out_path, index=False)

    print(f"\n=== Summary ===")
    print(f"  Plants queued : {len(queue_plant_ids)}")
    print(f"  Queue rows    : {len(queue)} "
          f"({len(cp)} candidate_pick, {len(cr)} confirm_reported)")
    print(f"\nWritten: {out_path}")
    print(f"\nNOTE: 'stage2a_score' in this file is Stage 2a's output, standing in "
          f"for the plan's 'stage2b_score' -- see module docstring.")


if __name__ == "__main__":
    main()