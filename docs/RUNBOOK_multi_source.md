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
| 01e holdout, both bins | 2026-11-06 | job 7608974 did nothing: sbatch --export split BINS="corrections,correct" at the comma, so it ran --holdout-bins corrections and resumed all 560. Fixed: 01e and 12 slurm turn '+' or spaces into commas (BINS="corrections+correct", STATES="OH+WI+KS") |
| 01e holdout rerun | 2026-11-06 | job 7608990, exit 0: --holdout-bins corrections,correct; 680 candidates / 34 plants, 560 resumed, 120 new, 0 failed. Only 6 of the 300 correct-bin plants were flagged to Stage 2 (out of sample) |
| 05b (holdout OD only) | 2026-11-06 | job 7609018, exit 0: OD for 680/127,488 candidates (34 holdout plants; the <50% warning is expected before the full 01e array). Fallback to Stage 2a for 6,355 plants = 6,345 without OD + 10 holdout plants where no candidate fired (by design); 24 plants re-ranked, 10 with a new #1 (median promoted from Stage 2a rank 6) |
| 12 (multi) | 2026-11-06 | job 7609019, exit 0. Corrections bin END-TO-END right 34/48 = 70.8% (Wilson 57-82%): Stage 1 accepted 20 (17 right, precision 85%), Stage 2 moved 17 of 28 right; re-rank @1 17/23 vs Stage 2a 13/23 (+4 plants); truth in the top 20 for 23/28 flagged. Correct bin: only 6/281 flagged (2.1%), but Stage 1 also accepted a wrong site for 7, so the end-to-end false-move rate is 12/281 = 4.3% (Wilson 2.5-7.3%), not the 2.1% the old section printed. 12 now reports that, the false accepts by sources, and OSM-passed plants by outcome (the old OSM line assumed the rule keeps the reported parcel, which is single-seed logic). Next: the single-seed baseline on the same holdout |
| 12 rerun | 2026-11-06 | job 7609022 with e3eb9ab: FALSE-MOVE RATE 12/281 = 4.3% (>1k pop 4.8%, 100-1k 2.0%). Of the 7 false accepts on correct plants, 4 are sites without the CWNS point (real moves) and 3 hold it (likely a truth/seed parcel mismatch next door, so 3.2-4.3%). 4 of the 7 came through the OSM rule, which is 202/207 = 97.6% right across both bins |
| 12 single-seed baseline | 2026-11-06 | job 7609024 (TPQA_SEED_MODE=cwns_only; pushed by hand, autopush is multi-only), same 414-plant holdout. Corrections bin right at #1: 21/50 = 42.0% (Wilson 29-56%) vs multi-source 34/50 = 68.0% (54-79%) counting multi's 2 unscored as wrong, 34/48 = 70.8% scored: +13 plants. Correct bin false moves: 10/281 = 3.6% (single-seed flags always move, the reported parcel being excluded) vs multi 12/281 = 4.3%: the same within noise. Single-seed has the larger candidate pool (truth in top 20 for 28/50 vs 23/50) and the stronger re-rank gain (+9 @1); multi wins by Stage 1 fixing 17 plants outright (10 of the 11 OSM-passed corrections among them) |
| 01e full array | 2026-11-06 | job 7609031: 48/48 exit 0 (RI has no candidates). 126,797 processed + 680 resumed of 127,488; 920 every-tile-failed (0.7%; mostly HTTP 403 on a whole plant's 20 candidates at once, plus ~50 TIFF read errors) and 11 'prepare failed' after STAC rate-limit retries ran out. Worst: AR 142, MI 125, MO 89, NY 71, NC 70. Resubmit the same array: resume retries only od_ran False rows and the 11 never written |
| 01e full array rerun | 2026-11-06 | job 7609222: 48/48 exit 0. 931 processed + 126,557 resumed = 127,488: every candidate now has detection results, 0 every-tile-failed, 0 'prepare failed'. The 403s did not repeat, so they were transient, not forbidden blobs (likely signed URLs expiring in the thread-local dataset cache over the long first run; a short rerun clears them). Only ~20 STAC rate-limit retries, all recovered |
| 05b (full coverage) | 2026-11-06 | job 7609276: detection results on 127,488/127,488 candidates (100%, was 0.5%). Fallback to Stage 2a ordering for 2,028 of 6,379 plants (32%; nothing fired in their top 20, close to 06b's 36% of true parcels with no detection). 1,328 plants (20.8%; 31% of the 4,351 where something fired) get a new #1; median Stage 2a rank of the promoted candidate 3. Holdout plants already had full detection results, so 12 is unchanged and not rerun |
| 13 (CUTOFF=0.95) | 2026-11-06 | job 7609280, exit 0. 17,874 plants: moved 3,052 (17.1%; 2,297 Stage 1 accepted a non-CWNS seed site, 755 Stage 2 moves), kept_model 3,910, flagged_not_moved 4,609, kept_site 562, kept_osm 306, reviewer 3,117, not_assessed 2,318. Of 1,317 flagged plants clearing 0.95, 755 move and 562 keep a site holding the reported parcel. Move distance median 372 m, 90th pct 2.0 km; 105 moves > 5 km and 19 > 25 km (max 420 km, TX Hunter's Glen), almost all Stage 1 seed-site accepts; 63 moves > 5 km lack CWNS's own address within 1 km. Every seed-site move has a source within 1 km; the 142 with none are Stage 2 moves near the CWNS point. Highest move rates MI 38.8%, IL 35.2%. Viewer export written to diagnostics/output_multi/, but viewer/ still defaults to diagnostics/output/ (the Oct 26 single-seed run) |
| decision: long moves | 2026-11-06 | Owner: keep the moves. The 63 moves > 5 km without CWNS's own address within 1 km stay as moved; no distance cap or review hold on 13's output |
| viewer -> output_multi | 2026-11-06 | viewer/ now follows TPQA_SEED_MODE like the pipeline (default multi -> diagnostics/output_multi/; cwns_only -> diagnostics/output/, the Oct 26 single-seed run). Checked locally: 13 output 17,874 plants (moved 3,052), 15,474 sites, 42,394 detections; explore 127,488 candidates / 6,379 plants, 85,851 objects. 13 and export_viewer_data now name the folder they actually wrote |
| 12 per-plant file | 2026-11-06 | 12 now also writes correction/diagnostics/holdout_plants_<seed mode>.csv: one row per holdout plant with right/wrong, outcome and scored, the same 'right' its headline uses (no move cutoff in either mode). Rerun 12 in both modes to pair single-seed and multi-source plant by plant (McNemar on the plants that changed) |
| 12 paired, single-seed vs multi | 2026-11-09 | jobs 7624559 (cwns_only) and 7623455 (multi), per-plant files in correction/diagnostics/. Both reproduce their earlier totals (21/50 vs 34/50; 10/281 vs 12/281). Misplaced plants: multi fixed 14 that single-seed got wrong and lost 1 (06004010001, a Stage 1 false accept): exact McNemar p = 0.001. The gain is mostly STAGE 2: 10 of the 14 were put right by multi's Stage 2, 4 by Stage 1 accepting another source's site. Of multi's 17 Stage 1 accept-rights, 13 were already right in single-seed through Stage 2, so the 2026-11-06 row's 'multi wins by Stage 1 fixing 17 plants outright' overstates Stage 1. Correct plants: 10 differ, 4 better / 6 worse, p = 0.75, no detectable change in false moves. Figures: docs/location_figures_20261109/ (fig10, fig11) |
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
