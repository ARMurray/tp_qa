"""
09_build_holdout.py
=====================
Samples the frozen evaluation holdout ONCE and writes both the manifest that
every training script anti-joins against and the truth table it is scored
against.

DRAWN FROM LABELED PLANTS ONLY (decided 2026-08-24)
----------------------------------------------------
The pool is the training gpkg's labeled universe, not all ~16,400 treatment
plants. Labeled plants already carry their own ground truth, so the holdout is
scoreable immediately instead of after a few hundred manual reviews:

  corrections bin        true location = Corrected_X/Y            derivable
  classes bin, Correct   true location = the reported point       derivable
  classes bin, Incorrect true location is UNKNOWN                 needs review

Only the third group needs a human. It is worth including anyway -- "the
reported point is wrong and we do not yet know the answer" is exactly the
situation the deployed pipeline faces -- but it is sampled small, and its plants
score as unknown rather than counting against the model until reviewed.

THREE SEPARATE METRICS, NOT ONE
--------------------------------
These bins measure different things and averaging them hides both:

  corrections  recall@1: of plants known to be misplaced, how often does the
               pipeline land on the right parcel. The headline number, and what
               Stage 2b exists to move.
  correct      false-move rate: of plants already correct, how often does the
               pipeline wrongly relocate them. A real harm, invisible in the
               metric above, and the thing that degrades if the model is tuned
               only on corrections.
  incorrect    review-dependent; reported separately once truth exists.

Hence explicit per-bin sizes rather than a single --n. Sampling proportionally
from the labeled pool would put only ~35 corrections in the holdout, a standard
error near 8pp -- too noisy to detect the improvements this loop exists to
produce.

THE COST, STATED PLAINLY
-------------------------
Every correction reserved here is one Stage 2b never trains on, and there are
only 316. --n-corrections 50 leaves 266. That hurts now and is still right: an
untrustworthy metric means you cannot tell whether any later round helped, and
the review loop refills training data while the holdout stays fixed by design.

GROWING THE HOLDOUT LATER
--------------------------
Re-sampling is forbidden (see the guard below). APPENDING is not: adding plants
that have never been trained on is safe, and is how the corrections slice gets
less noisy over time. Use --append-cohort. Every row carries a `cohort` label so
12_score_holdout.py can report cohort A separately -- that series stays
comparable across every round, while the all-cohort number gets tighter.

THE TRUTH FILE IS DERIVED, NOT ACCUMULATED
-------------------------------------------
holdout_truth.parquet is filtered to manifest membership immediately before it
is written, under every code path. A truth row for a plant that is not in the
current manifest is not evaluation data -- that plant is in TRAINING, and
scoring it would inflate recall@1 with plants the model was fit on.

This matters because the file is unioned with its previous contents so that
--append-cohort accumulates, and so that review-supplied truth for the
unlabeled bin survives a rerun. Without the membership filter, a
--force-resample leaves behind every superseded draw's rows (observed
2026-08-24: a 425-plant manifest paired with a 522-row truth file, 172 of them
labeled plants that had already re-entered training).

Usage:
    python 09_build_holdout.py --dry-run
    python 09_build_holdout.py
    python 09_build_holdout.py --append-cohort B --n-corrections 30 --n-correct 0
"""
import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from holdout import HOLDOUT_DIR, MANIFEST_PATH, TRUTH_PATH, manifest_checksum

TRUTH_SCHEMA = ["CWNS_ID", "true_ll_uuid", "true_lon", "true_lat",
                "truth_source", "reviewed_at", "reviewer"]


def load_labeled_pool() -> pd.DataFrame:
    """Labeled plants with their bin, true coordinates where derivable, and
    STATE_CODE from CWNS.

    A plant in both layers is treated as a correction -- the corrections layer
    is the more specific and more recent statement about it."""
    corrections = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CORRECTIONS)
    corrections["CWNS_ID"] = corrections["CWNS_ID"].astype(str)
    corrections = corrections.dropna(subset=["Corrected_X", "Corrected_Y"])
    corr = pd.DataFrame({
        "CWNS_ID": corrections["CWNS_ID"],
        "bin": "corrections",
        "true_lon": pd.to_numeric(corrections["Corrected_X"], errors="coerce"),
        "true_lat": pd.to_numeric(corrections["Corrected_Y"], errors="coerce"),
        "truth_source": "corrections_layer",
    }).dropna(subset=["true_lon", "true_lat"])

    classes = gpd.read_file(C.TRAINING_GPKG, layer=C.TRAINING_LAYER_CLASSES)
    classes["CWNS_ID"] = classes["CWNS_ID"].astype(str)
    print(f"  'class' column values: {classes['class'].value_counts().to_dict()}")

    cls = pd.DataFrame({"CWNS_ID": classes["CWNS_ID"], "class": classes["class"]})
    cls["bin"] = np.where(cls["class"].eq("Correct"), "correct", "incorrect")
    cls = cls.drop(columns=["class"])
    cls = cls[~cls["CWNS_ID"].isin(set(corr["CWNS_ID"]))]

    pool = pd.concat([corr, cls], ignore_index=True).drop_duplicates(subset="CWNS_ID")

    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt",
                      dtype={"CWNS_ID": str}, encoding="latin1")
    loc["LATITUDE"] = pd.to_numeric(loc["LATITUDE"], errors="coerce")
    loc["LONGITUDE"] = pd.to_numeric(loc["LONGITUDE"], errors="coerce")
    loc = loc.dropna(subset=["LATITUDE", "LONGITUDE"]).drop_duplicates(subset="CWNS_ID")
    pool = pool.merge(loc[["CWNS_ID", "STATE_CODE", "LATITUDE", "LONGITUDE"]],
                      on="CWNS_ID", how="inner")

    # Correct-bin truth IS the reported point -- that is what the label asserts.
    is_correct = pool["bin"].eq("correct")
    pool.loc[is_correct, "true_lon"] = pool.loc[is_correct, "LONGITUDE"]
    pool.loc[is_correct, "true_lat"] = pool.loc[is_correct, "LATITUDE"]
    pool.loc[is_correct, "truth_source"] = "reported_correct"
    pool.loc[pool["bin"].eq("incorrect"), "truth_source"] = "needs_review"
    return pool.reset_index(drop=True)


def load_unlabeled_pool(labeled_ids: set[str]) -> pd.DataFrame:
    """Treatment plants with usable coordinates that carry NO label.

    These measure something the labeled bins structurally cannot: the
    DEPLOYMENT distribution. The labeled pool is not a random sample of
    treatment plants -- somebody chose which ones to verify, and that choice may
    correlate with state, plant size, or how suspicious the coordinates looked.
    So the labeled bins' 87% correct rate is not necessarily the national rate,
    and no amount of labeled holdout will reveal the difference.

    A random draw from the unlabeled universe does. It is the only slice that
    answers "what will this pipeline actually do across all ~16,400 plants",
    and it is the only one that can surface a failure mode absent from the
    labeled data entirely.

    Cost: every one needs review before it scores. That is a one-time cost --
    once truth is recorded these plants are scored by join forever after, same
    as the labeled ones."""
    facility_types = pd.read_csv(C.CWNS_DIR / "FACILITY_TYPES.txt",
                                 dtype=str, encoding="latin1")
    treatment_ids = set(
        facility_types.loc[facility_types["FACILITY_TYPE"] == "Treatment Plant", "CWNS_ID"])

    loc = pd.read_csv(C.CWNS_DIR / "PHYSICAL_LOCATION.txt",
                      dtype={"CWNS_ID": str}, encoding="latin1")
    loc = loc[loc["CWNS_ID"].isin(treatment_ids)].copy()
    loc["LATITUDE"] = pd.to_numeric(loc["LATITUDE"], errors="coerce")
    loc["LONGITUDE"] = pd.to_numeric(loc["LONGITUDE"], errors="coerce")
    loc = loc.dropna(subset=["LATITUDE", "LONGITUDE"]).drop_duplicates(subset="CWNS_ID")
    loc = loc[~loc["CWNS_ID"].isin(labeled_ids)]

    out = loc[["CWNS_ID", "STATE_CODE", "LATITUDE", "LONGITUDE"]].copy()
    out["bin"] = "unlabeled"
    out["true_lon"] = np.nan
    out["true_lat"] = np.nan
    out["truth_source"] = "needs_review"
    return out.reset_index(drop=True)


def lookup_true_parcels(pool: pd.DataFrame) -> pd.DataFrame:
    """Resolve each derivable truth coordinate to its containing parcel.

    Opportunistic -- a null true_ll_uuid is fine. 12_score_holdout.py's primary
    test is whether the top-1 candidate parcel CONTAINS the true point, which is
    robust to parcel-id churn between Regrid vintages in a way that comparing
    ll_uuid strings is not. The uuid is stored because it makes scoring a cheap
    join whenever the vintage has not moved."""
    have = pool[pool["true_lon"].notna()].copy()
    if not len(have):
        pool["true_ll_uuid"] = None
        return pool

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false;")
    frames = []
    failed_states: list[tuple[str, int, str]] = []
    for state, grp in have.groupby("STATE_CODE"):
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
            # Non-fatal by design -- coordinate containment is the primary
            # scoring test -- but NOT silent. A read error here degrades truth
            # resolution for an entire state, and a run that quietly resolves
            # fewer parcels than the last one looks identical in the log.
            failed_states.append((str(state), len(grp), str(e)[:200]))
            print(f"    [{state}] parcel lookup FAILED for {len(grp)} plants "
                  f"(non-fatal): {str(e)[:200]}")
        finally:
            con.unregister("pts")
    con.close()

    matched = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=["CWNS_ID", "true_ll_uuid"])
    pool = pool.merge(matched, on="CWNS_ID", how="left")
    n_derivable = int(pool["true_lon"].notna().sum())
    n_resolved = int(pool["true_ll_uuid"].notna().sum())
    rate = 100 * n_resolved / n_derivable if n_derivable else 0.0
    print(f"  True parcel resolved for {n_resolved}/{n_derivable} "
          f"derivable-truth plants ({rate:.1f}%)")
    if n_derivable - n_resolved:
        print(f"    ({n_derivable - n_resolved} truth points fall on no parcel -- "
              f"scored by containment against coordinates instead)")

    if failed_states:
        n_lost = sum(n for _, n, _ in failed_states)
        print("\n  " + "!" * 68)
        print(f"  WARNING: parcel lookup errored in {len(failed_states)} state(s), "
              f"affecting {n_lost} plants.")
        print("  Those plants have no true_ll_uuid and will be scored by")
        print("  coordinate containment only. That is a valid fallback, but an")
        print("  ERRORED lookup is not the same as a point genuinely on no")
        print("  parcel -- rerun before trusting a resolution-rate comparison")
        print("  against an earlier run.")
        for st, n, msg in failed_states:
            print(f"    {st}: {n} plants -- {msg}")
        print("  " + "!" * 68 + "\n")

    return pool


def sample_bin(pool: pd.DataFrame, bin_name: str, n: int, seed: int) -> pd.DataFrame:
    """Draw n plants from one bin, spread across states by proportional
    allocation so the slice is not concentrated in whichever states dominate
    that bin."""
    cell = pool[pool["bin"] == bin_name]
    if n <= 0 or not len(cell):
        return cell.head(0)
    if n >= len(cell):
        print(f"  WARNING: requested {n} from '{bin_name}' but only {len(cell)} "
              f"exist -- taking all of them, which leaves NONE for training")
        return cell.copy()

    frac = pool[pool["bin"] == bin_name].groupby("STATE_CODE").size()
    exact = frac / frac.sum() * n
    alloc = np.floor(exact).astype(int).clip(lower=0, upper=frac)
    for st in (exact - np.floor(exact)).sort_values(ascending=False).index:
        if alloc.sum() >= n:
            break
        if alloc[st] < frac[st]:
            alloc[st] += 1
    while alloc.sum() > n:
        alloc[alloc.idxmax()] -= 1

    # NOT hash(bin_name): str hashing is salted by PYTHONHASHSEED and varies
    # per process, so --seed silently failed to reproduce a draw across runs
    # (observed 2026-08-24: a dry run and the committed run with the same seed
    # produced different samples). sha256 is stable forever.
    bin_salt = int.from_bytes(hashlib.sha256(bin_name.encode()).digest()[:4],
                              "big") % 10_000
    rng = np.random.default_rng(seed + bin_salt)
    picks = [cell[cell["STATE_CODE"] == st].sample(
                n=int(k), random_state=int(rng.integers(0, 2**31 - 1)))
             for st, k in alloc.items() if k > 0]
    return pd.concat(picks, ignore_index=True) if picks else cell.head(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-corrections", type=int, default=50,
                    help="known-misplaced plants. The headline recall@1 slice. "
                         "Each is a correction Stage 2b will not train on.")
    ap.add_argument("--n-correct", type=int, default=150,
                    help="known-correct plants, for the false-move rate. Cheap: "
                         "there are ~2000 and Stage 1 has plenty.")
    ap.add_argument("--n-incorrect", type=int, default=0,
                    help="known-wrong-but-unresolved. Normally 0: as of "
                         "2026-08-24 every Incorrect plant in the classes layer "
                         "also has a correction, so this bin is empty.")
    ap.add_argument("--n-unlabeled", type=int, default=75,
                    help="randomly drawn UNLABELED plants -- the deployment "
                         "distribution, and the only slice that estimates the "
                         "true national error rate. Each needs a one-time "
                         "review before it scores.")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--states", type=str, default=None)
    ap.add_argument("--cohort", type=str, default="A")
    ap.add_argument("--append-cohort", type=str, default=None,
                    help="add a NEW cohort without disturbing the existing "
                         "holdout. The legitimate way to grow it.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-resample", action="store_true",
                    help="overwrite the manifest. DESTROYS comparability with "
                         "every score recorded so far. Use --append-cohort "
                         "instead unless you truly mean this.")
    args = ap.parse_args()

    C.ensure_dirs()
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 09_build_holdout.py ===\n")

    appending = args.append_cohort is not None
    cohort = args.append_cohort if appending else args.cohort
    existing_manifest = pd.read_parquet(MANIFEST_PATH) if MANIFEST_PATH.exists() else None

    if existing_manifest is not None and not appending and not args.force_resample:
        print(f"Manifest already exists: {MANIFEST_PATH}")
        print(f"  {len(existing_manifest)} plants, cohorts "
              f"{sorted(existing_manifest['cohort'].unique())}, "
              f"first sampled {existing_manifest['sampled_at'].min()}")
        print(f"  checksum: {manifest_checksum()[:16]}...")
        print("\nRefusing to overwrite. The holdout is sampled ONCE by design --\n"
              "re-drawing rotates plants between train and eval and makes every\n"
              "previously recorded score incomparable.\n"
              "To GROW it safely:  --append-cohort B\n"
              "To discard history deliberately:  --force-resample")
        return

    if existing_manifest is not None and args.force_resample:
        print("*" * 72)
        print("WARNING: --force-resample overwrites the holdout manifest.")
        print("Every score recorded so far becomes incomparable. Previously held-out")
        print("plants may enter training; previously trained plants may enter the")
        print("holdout. The old manifest is backed up, but CONTAMINATION IS NOT")
        print("REVERSIBLE once models are retrained against the new split.")
        print("Consider --append-cohort instead.")
        print("*" * 72 + "\n")
        if not args.dry_run:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = MANIFEST_PATH.with_suffix(f".superseded-{stamp}.parquet")
            MANIFEST_PATH.rename(backup)
            print(f"Old manifest -> {backup.name}")
            # The truth file is filtered to manifest membership before writing,
            # so a resample drops the superseded draw's rows automatically.
            # Back them up first: they are real derived truth, and a later
            # --append-cohort that re-draws some of those plants would
            # otherwise have to re-resolve parcels it already had.
            if TRUTH_PATH.exists():
                tbackup = TRUTH_PATH.with_suffix(f".superseded-{stamp}.parquet")
                TRUTH_PATH.replace(tbackup)
                print(f"Old truth    -> {tbackup.name}")
            print()
            existing_manifest = None

    print("Loading labeled plant pool...")
    pool = load_labeled_pool()
    if args.states:
        keep = [s.strip() for s in args.states.split(",")]
        pool = pool[pool["STATE_CODE"].isin(keep)]
        print(f"  Restricted to {keep}")

    if appending and existing_manifest is not None:
        already = set(existing_manifest["CWNS_ID"].astype(str))
        pool = pool[~pool["CWNS_ID"].isin(already)]
        print(f"  Excluding {len(already)} plants already in the holdout")

    print(f"\nLabeled pool: {len(pool)} plants")
    for b, n in pool["bin"].value_counts().items():
        print(f"  {b:12s}: {n}")

    if args.n_unlabeled > 0:
        labeled_ids = set(pool["CWNS_ID"])
        unlabeled = load_unlabeled_pool(labeled_ids)
        if args.states:
            unlabeled = unlabeled[unlabeled["STATE_CODE"].isin(
                [s.strip() for s in args.states.split(",")])]
        if appending and existing_manifest is not None:
            unlabeled = unlabeled[~unlabeled["CWNS_ID"].isin(
                set(existing_manifest["CWNS_ID"].astype(str)))]
        print(f"Unlabeled pool: {len(unlabeled)} plants")
        pool = pd.concat([pool, unlabeled], ignore_index=True)

    print("\nSampling...")
    draw = pd.concat([
        sample_bin(pool, "corrections", args.n_corrections, args.seed),
        sample_bin(pool, "correct", args.n_correct, args.seed),
        sample_bin(pool, "incorrect", args.n_incorrect, args.seed),
        sample_bin(pool, "unlabeled", args.n_unlabeled, args.seed),
    ], ignore_index=True)

    n_corr_drawn = int((draw["bin"] == "corrections").sum())
    n_corr_pool = int((pool["bin"] == "corrections").sum())
    print(f"\n--- Draw: {len(draw)} plants (cohort {cohort}) ---")
    for b, n in draw["bin"].value_counts().items():
        print(f"  {b:12s}: {n}")
    print(f"  states      : {draw['STATE_CODE'].nunique()}")

    if n_corr_drawn:
        se = 100 * 0.5 / np.sqrt(n_corr_drawn)
        print(f"\n  Corrections slice n={n_corr_drawn}: recall@1 standard error "
              f"about +/-{se:.1f}pp (worst case p=0.5).")
        print(f"  Leaves {n_corr_pool - n_corr_drawn} corrections for training "
              f"(pool was {n_corr_pool}).")
        if se > 8:
            print("  That error bar is wide enough that small round-over-round moves\n"
                  "  will not be distinguishable from noise. Either raise\n"
                  "  --n-corrections now, or plan to --append-cohort once the review\n"
                  "  loop has produced more corrections.")

    n_unlab = int((draw["bin"] == "unlabeled").sum())
    if n_unlab:
        print(f"\n  Unlabeled slice n={n_unlab}: national error-rate estimate "
              f"carries a standard error of about "
              f"+/-{100 * 0.5 / np.sqrt(n_unlab):.1f}pp.")
        print(f"  These need a one-time review before they score. Review them")
        print(f"  FIRST in round 1 -- until they have truth, the deployment-")
        print(f"  distribution number does not exist.")

    print("\nResolving true parcels...")
    draw = lookup_true_parcels(draw)
    needs_review = int(draw["truth_source"].eq("needs_review").sum())
    print(f"  Truth already known : {len(draw) - needs_review}  "
          f"(labeled bins -- scoreable immediately)")
    print(f"  Needs review        : {needs_review}  "
          f"(unlabeled bin -- one-time cost)")

    manifest = draw[["CWNS_ID", "STATE_CODE", "bin"]].copy()
    manifest["stratum"] = draw["bin"] + "|" + draw["STATE_CODE"].astype(str)
    # Unlabeled plants were never in any training bin; labeled ones were. This
    # is what lets 12_score_holdout.py report the two populations separately,
    # which is the whole point of including both.
    manifest["in_original_training"] = draw["bin"].ne("unlabeled")
    manifest["cohort"] = cohort
    manifest["sampled_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["sample_seed"] = args.seed

    truth = draw[draw["truth_source"].ne("needs_review")].copy()
    truth_out = pd.DataFrame({
        "CWNS_ID": truth["CWNS_ID"],
        "true_ll_uuid": truth["true_ll_uuid"] if "true_ll_uuid" in truth.columns else None,
        "true_lon": truth["true_lon"],
        "true_lat": truth["true_lat"],
        "truth_source": truth["truth_source"],
        "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        "reviewer": "derived_from_training_gpkg",
    })[TRUTH_SCHEMA]

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    if appending and existing_manifest is not None:
        manifest = pd.concat([existing_manifest, manifest], ignore_index=True)
        print(f"\nAppended cohort {cohort}: "
              f"{len(existing_manifest)} -> {len(manifest)} plants")
    manifest.to_parquet(MANIFEST_PATH, index=False)
    print(f"\nWritten: {MANIFEST_PATH}  ({len(manifest)} plants)")
    print(f"  checksum: {manifest_checksum()}")

    # Union with whatever is already there -- this is what lets --append-cohort
    # accumulate, and what preserves review-supplied truth for the unlabeled bin
    # across reruns. keep="last" so a freshly derived row wins over a stale one;
    # the reverse silently pinned truth to whichever run happened to be first.
    if TRUTH_PATH.exists():
        prev = pd.read_parquet(TRUTH_PATH)
        if len(prev):
            prev["CWNS_ID"] = prev["CWNS_ID"].astype(str)
            truth_out = pd.concat([prev, truth_out], ignore_index=True) \
                          .drop_duplicates(subset="CWNS_ID", keep="last")

    # THEN filter to manifest membership. This is the invariant that makes the
    # truth file safe to score directly: every row in it is a plant that is
    # excluded from training. Runs under BOTH paths -- on --append-cohort the
    # manifest already includes the earlier cohorts, so nothing legitimate is
    # dropped; on --force-resample the superseded draw's rows fall away here,
    # which is the whole point. Not delegated to 12_score_holdout.py, because
    # then the guarantee depends on 12 remembering to apply it.
    in_manifest = set(manifest["CWNS_ID"].astype(str))
    before = len(truth_out)
    truth_out = truth_out[truth_out["CWNS_ID"].astype(str).isin(in_manifest)].copy()
    dropped = before - len(truth_out)
    if dropped:
        print(f"\n  Dropped {dropped} truth row(s) for plants not in the current "
              f"manifest.")
        print(f"  Those plants are in TRAINING now -- scoring them would inflate")
        print(f"  recall@1 with plants the model was fit on.")

    truth_out.to_parquet(TRUTH_PATH, index=False)
    n_scoreable = len(truth_out)
    print(f"Written: {TRUTH_PATH}  ({n_scoreable} plants with truth)")
    if n_scoreable > len(manifest):
        # Should be unreachable given the filter above; if it ever fires the
        # invariant has been broken by an edit, and that is worth a hard stop.
        raise RuntimeError(
            f"truth rows ({n_scoreable}) exceed manifest plants ({len(manifest)}) "
            f"-- the membership filter is not doing its job")

    print("\nNEXT:")
    print("  1. Apply PATCH_holdout_antijoins.md to 03/04/06/07 -- until then the")
    print("     manifest protects nothing.")
    print(f"  2. Retrain. Stage 2b loses ~{n_corr_drawn} plants' worth of rows.")
    print(f"  3. Pilot inference, then 12_score_holdout.py: {n_scoreable} of the")
    print(f"     {len(manifest)} holdout plants score immediately; the "
          f"{needs_review} unlabeled")
    print("     ones need a one-time review first.")
    print("\n=== complete ===")


if __name__ == "__main__":
    main()