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

Every job already copies its log to `correction/diagnostics/latest_logs/`. In multi mode each job also refreshes
**`correction/diagnostics/multi_status.txt`** when it ends: one file with, in run order, the window sizes against the
single-seed run, seed -> parcel matching, seed-site detection coverage, the Stage 1 / Stage 2 tables, thresholds and
top features, 05's decisions, 13's provenance, and the last 60 lines of any job whose exit status was not 0.

Getting it back through git is one command on a login node, run after a step finishes:

```bash
bash /work/GRDVULN/tp_qa/correction/scripts/push_diagnostics.sh "01a OH done"
```

It regenerates the status file, commits only `correction/diagnostics/`, and pushes. Then `git pull` anywhere and read
`multi_status.txt`. It never touches data, models or code.

## Order

Submit each step after the previous one has finished (see [05_RUNBOOK.md](05_RUNBOOK.md) A2-A6 for why).
`cd /work/GRDVULN/tp_qa && git pull && cd correction/scripts`

```bash
# 1. labels (unchanged)
sbatch 00_build_training_bins.slurm

# 2. NEW, in this order. All three take the FULL universe: the Stage 1 table only uses labelled plants, but
#    inference needs every plant, and 01f/01e resume per state / per parcel.
sbatch --array=0-50%6 --export=FULLUNIVERSE=1,NORESUME=1 01a_extract_parcels.slurm      # nlcd_features_multi/
sbatch --array=0-50%6 --export=FULLUNIVERSE=1,NORESUME=1 01f_build_seed_sites.slurm     # seed_sites/
sbatch --array=0-47%6 --export=SEEDSITES=1 01e_run_od_candidates_array.slurm            # od_features_seed_sites/
#    01d top-up is only needed if 08 shows parcels missing from 01a; it appends to nlcd_features_multi/.
#    01e: about 24.6k sites nationally, ~4.1k for the labelled plants. At the measured 39 candidates/min that
#    is ~10.5 h aggregate (1.7 h for labelled plants alone). Rate-limited, not CPU-limited: see the array wrapper.

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

## What has been tested, and what has not

Tested on synthetic parcel stores (`correction/tests/`, all pass locally): seed -> parcel -> site, the Stage 1 site table
and its labels (including that no label-derived column reaches the model), 03 on a synthetic table, the Stage 2
window and seed features, 05's Stage 1 decision and Stage 2a pool, 13's provenance, 12's outcomes, 01e's scope
(detector stubbed), the path switch.

**Not run on real data** (the parcels, NAIP and CWNS exports are HPC-only): every step above on the real Regrid store.
Expect first-run surprises in 01f's point-in-parcel join, 02's real-data joins, and run time. The recall check
(`check_seed_recall.py`) was real data at the H3 level.

Known limits to keep in mind when reading results:

- **Stage 2 is trained on corrected plants only**, where the CWNS point's parcel is wrong by construction, so it learns to
  rank the CWNS-point parcel low. At inference that parcel is only reached when Stage 1 declined it, which is the same
  population the old pipeline excluded outright. Watch the holdout `correct` bin for plants wrongly moved.
- **Stage 1 sites with no detection** (AK/HI/PR have no NAIP) get `od_ran` False; outside those states a missing
  detection is a gap to fill, not a feature.
- **Seeds on no parcel** within 100 m stay unparcelled: they inform Stage 2's distance features but form no site.
- The plant-level accept threshold is calibrated on the held-out 25% of plants, so it is noisy; the holdout comparison is the
  real test.
