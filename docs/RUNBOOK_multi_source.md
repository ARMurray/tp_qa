# Running the multi-source pipeline on the HPC

Written 2026-11-03. The design is in [PLAN_20261103_multi_source_seeds.md](PLAN_20261103_multi_source_seeds.md);
this is the order to run it in, what to check, and what has and has not been tested.

## What changed

Every plant-role location (CWNS point, CWNS geocode, ICIS-NPDES) now seeds the pipeline, not just the CWNS point.

| Stage | Now |
|---|---|
| seeds | `correction/data/candidates/plant_seeds.csv` (tracked in git; `build_plant_seeds.py` makes it) |
| 01a | each seed is a window center; writes `nlcd_features_multi/` |
| **01f** (new) | seed -> parcel -> **seed site**; seeds on one parcel, or on neighbouring parcels with one owner, collapse into one site. Writes `data/seed_sites/state=XX/` |
| **01e `--seed-sites`** | detection on each site's representative parcel. Writes `od_features_seed_sites/` |
| 02 | Stage 1 table has one row **per site**, labelled by whether the site holds the truth parcel; Stage 2 window and per-candidate seed features |
| 03 | trains on sites with plant-grouped splits; accept threshold calibrated on the **plant-level** outcome |
| 04 | unchanged code; the seed features are just more columns |
| 05 | scores every site; accepts a plant when its best site clears the threshold, else Stage 2 on the union of the seed windows (nothing excluded) |
| 13 | provenance columns; a Stage 1 accept of a site without the CWNS point is placed like a move |
| 12 | end-to-end outcomes that count Stage 1 accepts |

`TPQA_SEED_MODE=multi` is the default. Everything the seeds change goes to a `*_multi` directory (`models_multi/`,
`features_multi/`, `inference_multi/`, `output_multi/`, `review_queue_multi/`, ...), so the single-seed pipeline's files are
untouched. `TPQA_SEED_MODE=cwns_only` reproduces the old behaviour and old paths (`export TPQA_SEED_MODE=cwns_only` before
any command). Shared and never suffixed: CWNS exports, training labels, the frozen holdout, 01b/01c detection, review logs,
`best.pt`.

## Reading results without HPC access

Nothing to run by hand. When any job ends it (1) copies its log to `correction/diagnostics/latest_logs/`, (2) refreshes
**`correction/diagnostics/multi_status.txt`** -- one file with, in run order, the window sizes against the single-seed
run, seed -> parcel matching, seed-site detection coverage, the Stage 1 / Stage 2 tables, thresholds and top features,
05's decisions, 13's provenance, and the last 60 lines of any job that exited non-zero -- and (3) commits and pushes
`correction/diagnostics/` and nothing else (`_latest_log.sh`). A job array commits once, from its last task. So after a step
finishes, `git pull` anywhere and read the file.

The auto-push needs the compute node to reach GitHub with your login-node credentials. If a step's results do not show
up on GitHub, run the same thing by hand on a login node (and tell me, so we know the compute nodes cannot push):

```bash
bash /work/GRDVULN/tp_qa/correction/scripts/push_diagnostics.sh
```

`export TPQA_NO_AUTOPUSH=1` (`setenv` in tcsh) turns the auto-push off; the log copy and status file still happen.

## Order

Submit each step after the previous one has finished (see [05_RUNBOOK.md](05_RUNBOOK.md) A2-A6 for why).
`cd /work/GRDVULN/tp_qa && git pull && cd correction/scripts`

Check each step with `sacct -j <JOBID> -X --format=JobID%20,State,ExitCode` before submitting the next. A failed step
pushes its traceback too (see above).

```bash
# 1. labels (unchanged)
sbatch 00_build_training_bins.slurm

# 2. NEW, in this order. All three take the FULL universe: the Stage 1 table only uses labelled plants, but
#    inference needs every plant, and 01f/01e resume per state / per parcel.
#    First a one-state check (OH is array index 34): sbatch --array=34 ... 01a, read multi_status.txt, then the full array.
sbatch --array=0-50%6 --export=FULLUNIVERSE=1,NORESUME=1 01a_extract_parcels.slurm      # nlcd_features_multi/
sbatch --array=0-50%6 --export=FULLUNIVERSE=1,NORESUME=1 01f_build_seed_sites.slurm     # seed_sites/
sbatch --array=0-47%6 --export=SEEDSITES=1 01e_run_od_candidates_array.slurm            # od_features_seed_sites/
#    01d top-up is only needed if 08 shows parcels missing from 01a; it appends to nlcd_features_multi/.
#    01e volume: OH came out at 2.3 sites per plant (1,944 sites for 844 plants), so nationally about 40k sites, ~7k for the
#    labelled plants. At the measured 39 candidates/min that is ~17 h aggregate (~3 h for labelled plants alone).
#    Rate-limited, not CPU-limited: see the array wrapper.

# 3. features. 02 refuses to build the Stage 1 table without seed-site detections (--no-seed-od overrides).
sbatch --export=FULL_UNIVERSE=1 02_feature_engineering.slurm
sbatch 02b_merge_feature_shards.slurm

# 4. models
sbatch 03_train_stage1.slurm        # models_multi/stage1_*; read the "Plant-level accept analysis" block
sbatch 04_train_stage2.slurm

# 5. inference, then the rest exactly as in 05_RUNBOOK.md steps 5-7b (they now read and write *_multi/)
sbatch --array=0-39,41-47%8 05_run_inference_array.slurm
sbatch --array=40 --mem=128G 05_run_inference_array.slurm
sbatch merge_05_shards.slurm
sbatch --export=SCOPE="train" 01e_run_od_candidates.slurm      # re-ranker training detections
sbatch 06b_build_rerank_training.slurm
sbatch 07b_train_rerank.slurm
sbatch --array=0-47%6 01e_run_od_candidates_array.slurm        # detection on every candidate
sbatch 01e_run_od_candidates.slurm                             # holdout scope
sbatch 05b_rerank_candidates.slurm
sbatch 12_score_holdout.slurm
sbatch --export=CUTOFF=0.95 13_build_corrected_output.slurm
```

To score the old pipeline for the comparison, run `12_score_holdout` with `TPQA_SEED_MODE=cwns_only`: it reads the
original `inference/` and `models/`, which are still there. Compare **END-TO-END right** in the multi-source block against
the old recall@1 on the same holdout.

## Check as you go

- **01a**: window center count printed (`Window centers after adding plant seeds`); parcels per state should rise modestly
  over the single-seed run (the recall check put the median plant window at 1.07x a single disk).
- **01f**: `Seed -> parcel:` line per state. `none` should be a small share (seeds on no parcel and beyond the 100 m snap).
  `sites per plant` mostly 1-2.
- **01e seed sites**: `Candidates processed` per state; resume skips finished parcels.
- **02**: `Labelled plants`, `Correct sites per plant` (should be mostly 0 or 1; a 2 is a plant with two sites holding
  parts of the truth), `Seed-site OD joined: od_ran True for N / M` (should be close to M outside AK/HI/PR).
- **03**: refuses to train if a feature looks like the label (`find_label_leaks`). Then the plant-level table: pick the
  precision you can live with. `ACCEPT THRESHOLD` is written to `stage1_accept_threshold.joblib`.
- **05**: `Accepted by Stage 1`, `OSM rule`, `multi_pass flagged`. A large multi_pass count means two sources disagree
  about a lot of plants.
- **13**: `location basis:` and `moved plants led by:` -- the answer to "which source found it".

## Provenance in the output (`output_multi/cwns_corrected_locations.*`)

| Column | Meaning |
|---|---|
| `loc_basis` | `reviewer`, `stage1_accept` (the CWNS point's site), `stage1_accept_seed` (a site without the CWNS point), `stage2_move`, `kept_site`, `flagged_not_moved`, `not_assessed` |
| `loc_sources` | sources with a seed within 250 m of the output point, e.g. `cwns_point+icis_fac_frs` |
| `loc_n_origins` | independent origins among them (a copy of another seed counts once) |
| `nearest_source` | the closest source (ties within 1 m all listed); `led_by` is this, for moved plants only |
| `dist_to_final_<source>_m` | exact distance to each source, so the 250 m rule can change without a rebuild |
| `cwns_point_ll_uuid` | the parcel under the CWNS point (`reported_ll_uuid` is the parcel Stage 1 judged best) |
| `best_site_*`, `accepted_*`, `alt_site_*`, `multi_pass`, `n_sites` | the Stage 1 decision |

Source groups: `cwns_point` = CWNS coordinates, `cwns_addr` = CWNS geocode, `icis_fac_frs` / `icis_fac_addr` /
`icis_feat_INF` / `icis_feat_INO` = ICIS-NPDES.

## Run log (update as steps finish)

Full-universe run started 2026-11-03, walking the steps one at a time. Results are read from
`correction/diagnostics/multi_status.txt` after each step.

| Step | State | Result |
|---|---|---|
| 00 labels | done | training gpkg current |
| 01a, all 51 states | done 2026-11-03 | 85.6M parcels vs 77.7M single-seed; median state 1.09x. AK 5.5x and PR 2.7x (no NAIP there, no effect downstream). Largest of the rest: SC 1.40, DE 1.31, SD 1.29 |
| 01f, all 51 states | done 2026-11-03 | 55,945 seeds / 16,430 plants -> 34,482 sites. 613 seeds (1.1%) on no parcel, similar for every source. 81 plants (0.5%) have no site. Sites per plant: 1: 4,440, 2: 6,952, 3: 3,884, 4+: ~1,070. 18,203 sites lack the CWNS point (median 716 m from it; 58% within 1 km). Different-owner neighbours stay separate sites (6% of sites within 50 m of a sibling) |
| 01e `--seed-sites`, first pass | 2026-11-03, job 7606633 | 43 of 48 states exit 0 within ~6 h (not 17). CO, ID, NM, UT, WY OOM-killed (exit 137) before finishing a batch, so nothing was written: 01e had no oversized-parcel cap (01b's 2 km2 / 150 tiles) and flushed once per state. 32,193 / 34,482 sites covered (93.4%); od_ran False on 9.2% of rows (rate limiting), which resume counted as done |
| 01e fix | 2026-11-04 | cap as in 01b (one tile around the site's CWNS point, else an independent seed; `parcel_capped`, `parcel_cap_reason`), flush per 200-parcel batch, resume retries od_ran False rows. Resubmit the same array |
| 01e rerun | 2026-11-04 | all 48 tasks exit 0; 34,077 / 34,482 sites (98.8%; AK/HI/PR have no NAIP); 181 parcels capped; 2,641 still od_ran False. IL diagnostic: every failure is a 500 m tile crossing its NAIP quad's edge (the old fallback re-resolved the same quad). |
| 01e mosaic fix | 2026-11-04 | 01b `fetch_naip_mosaic`: a tile crossing a quad edge is filled from the neighbouring quads (same NAIP year first); fails only if under 50% covered. Resubmit the same array; resume retries only od_ran False rows |
| 01e mosaic rerun | 2026-11-04 | job 7607096: 48/48 exit 0, every task "Every tile failed: 0"; od_ran 100.0% (1 MN row False, no error recorded); detected 24.7% -> 26.7%, so the recovered parcels detect at the usual rate |
| 02, first pass | 2026-11-04 | two arrays ran at once (7607149, and 7607200 cancelled part-way); AK/HI/PR stopped on 'no seed-site detection' but reported exit 0, so 02b would have merged without them. Elsewhere OD joined 5,338 / 5,338 and correct sites per plant {1: 2560, 0: 282}. Fixed: no-NAIP states get od_ran False (C.NO_NAIP_STATES); the slurm script exits with Python's status. Rerun tasks 0,1,10,28,32,38,39, then 02b |
| 02 rerun + 02b | 2026-11-04 | 7 tasks rerun, all exit 0; AK/HI/PR od_ran False. 14_stage1 5,435 sites / 2,901 plants; 15_stage2 9.97M rows / 818 plants. CWNS-point site Correct 75.4%, other sites 16.8% |
| 03, first pass | 2026-11-04 | no label leak; spatial CV AUC 0.985; plant-level 95% target -> 0.673 (~85% captured). Raised STAGE1_ACCEPT_PRECISION to 0.975 (0.85 gave 97.7% / 67.6% captured on 323 held-out plants); rerun 03 |
| 03 rerun | 2026-11-05 | 97.5% target -> ACCEPT THRESHOLD 0.814 (Youden 0.554); same model, AUC 0.985 |
| 04, first pass | 2026-11-05 | 768 plants (14 with no positive), spatial CV AUC 0.9965, no errors -- but the 1:50 negative sample was drawn on parcel attributes only, so almost no negative sat on a seed parcel (0.02% of candidates) and the declined seed sites / CWNS-point parcel were barely in training. Fixed: 04 keeps every seed-parcel negative; rerun 04 |
| 04 rerun | 2026-11-05 | job 7608620, exit 0: 1,433 seed-parcel negatives kept within the same 37,700-negative budget; spatial CV AUC 0.9965 -> 0.9932, test AUC 0.9915, Youden 0.524 -> 0.329 (05 ranks by Stage 2 score and does not use this threshold). Parcel-size / ownership features now lead; on_seed_parcel and in_seed_site left the top 20 because sitting on a seed no longer means 'the plant' by itself |
| 05 + merge | 2026-11-05 | arrays 7608672 / 7608685 (task 40 at 128G): 48/48 exit 0. 15,337 plants; Stage 1 accepted 8,948 (58%; OSM rule 3,012), multi_pass 1; 6,389 to Stage 2 (73 no_parcel), 10 with no candidates. Low accept rates in NE/ND/WV/WI/NM/UT/NV match the states with many zero-correct training plants. Merge 7608730: 15,337 plant rows, 127,488 candidate rows (top 20 x 6,379), 32,615 site rows; RI has no Stage 2 plants. Log label fixed in a9419c6 (multi mode excludes no parcel) |
| 01e SCOPE=train | 2026-11-05 | job 7608823, exit 0, 106 min: 6,919 candidates for 346 plants, 1 failed (NH, TIFF read error), 37 rate-limit retries all recovered. Only 346 of 791 trainable corrections reach Stage 2 now: Stage 1 accepts the rest (in-sample, since it trained on them), so the re-ranker trains on a smaller, harder set |
| 06b | 2026-11-05 | job 7608843, exit 0: 6,714 rows / 336 plants (346 from 01e minus 10 whose true parcel did not resolve), 309 positives; every row has OD. Truth in top 20 for 92% (in-sample; the holdout was 69% single-seed), Stage 2a median rank 1. OD detection 64.4% on true parcels vs 5.7% on competitors |
| 07b | 2026-11-06 | job 7608860, exit 0: spatial CV AUC 0.974; on 84 held-out training plants (79 with the truth in the pool) recall@1 Stage 2a 60.8% -> re-rank 73.4% (+12.7pp), @3 86.1% -> 93.7%. Stage 2a rank/score lead, then osm_ww, seed distance and pool-relative OD. In-sample Stage 2a scores, so 12 is the real read. The ArrowStringArray shuffle warning was harmless (checked); silenced in e6b2aa3. Running holdout-scope 01e -> 05b -> 12 before the full 01e array, since 05b falls back to Stage 2a ordering for plants without OD and 12 scores holdout plants only |
| 01e holdout | 2026-11-06 | job 7608916, exit 0, 6 min: corrections bin 560 candidates / 28 plants, 0 failed. Stage 1 accepted the other 22 of the 50 corrections-bin plants (out of sample). 12 also scores the correct bin, and in multi mode a flagged correct plant can be moved_right (no parcel is excluded), so its order matters too: rerun with BINS=corrections,correct (resume skips the 560) before 05b |
| 02 onward | not started | |

Next, once 01e finishes: confirm all array tasks exited 0 (`sacct -j <JOBID> -X --format=JobID%20,State,ExitCode`; a
timed-out task resumes if resubmitted, since finished parcels are skipped), then 02 -> 02b -> 03 -> 04 -> 05 as in "Order".
At 02 read `Seed-site OD joined` (should be close to M outside AK/HI/PR) and `Correct sites per plant`; at 03 read the
plant-level accept table before choosing the precision.

## What has been tested, and what has not

Tested on synthetic parcel stores (`correction/tests/`, all pass locally): seed -> parcel -> site, the Stage 1 site table
and its labels (including that no label-derived column reaches the model), 03 on a synthetic table, the Stage 2
window and seed features, 05's Stage 1 decision and Stage 2a pool, 13's provenance, 12's outcomes, 01e's scope
(detector stubbed), the path switch.

**Run on real data so far**: 01a and 01f (all states; see the run log). **Not yet run on real data** (the parcels, NAIP and
CWNS exports are HPC-only): 01e seed-site detection and everything after it (02, 03, 04, 05, 13, 12). Expect first-run
surprises in 02's real-data joins and in run time. The recall check (`check_seed_recall.py`) was real data at the H3 level.

Known limits to keep in mind when reading results:

- **Stage 2 is trained on corrected plants only**, where the CWNS point's parcel is wrong by construction, so it learns to
  rank the CWNS-point parcel low. At inference that parcel is only reached when Stage 1 declined it, which is the same
  population the old pipeline excluded outright. Watch the holdout `correct` bin for plants wrongly moved.
- **Stage 1 sites with no detection** (AK/HI/PR have no NAIP) get `od_ran` False; outside those states a missing
  detection is a gap to fill, not a feature.
- **Seeds on no parcel** within 100 m stay unparcelled: they inform Stage 2's distance features but form no site.
- The plant-level accept threshold is calibrated on the held-out 25% of plants, so it is noisy; the holdout comparison is the
  real test.
