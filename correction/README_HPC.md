# correction/ on the HPC

Everything under `/work/GRDVULN/correction`. Object-detection model *training*
stays local (`tp_qa/detection/`); only the trained `best.pt` comes here.

## Layout

```
/work/GRDVULN/correction/
  .venv/                    built once by scripts/setup_env.sh
  scripts/                  config.py, model_utils.py, 01a-04, build_training_bins.py, *.slurm
  logs/                     all SLURM output
  models/                   stage1/stage2 model output
    object_detection/       best.pt uploaded from detection/
  data/
    cwns/                   PHYSICAL_LOCATION.txt, FACILITY_TYPES.txt,
                            DISCHARGES.csv, POPULATION_WASTEWATER.txt
    training/               Updates.gdb (uploaded), training_locations.gpkg (built)
    reference/              census gdb, Wastewater_Plants.gpkg
    nlcd_features/          01a output
    od_features/            01b output: tiles/ detections/ objects/ plants/
    features/               02 output
```

External, shared, **not** under this root:
- `/work/GRDVULN/data/parcels/` — Regrid parquet store
- `/work/GRDVULN/data/nlcd/` — NLCD raster

## First-time setup

```bash
bash /work/GRDVULN/correction/scripts/setup_env.sh
```

Then, before anything will run:

1. **Keyword lists are already filled in**, copied verbatim from the local
   `config.py`. Keep them in sync if the local lists ever change — they alter
   feature *values*, so a mismatch shows up as quietly different model results,
   not an error.
2. Upload the CWNS text exports to `data/cwns/`.
3. Upload `Updates.gdb` to `data/training/` and confirm `MASTER_LAYER` in
   config.py matches the current dated layer name.
4. Upload the census gdb and OSM gpkg to `data/reference/`. 02 runs without
   them but silently produces empty census/name-match features, which look
   like real negatives to the models.
5. Upload the trained `best.pt` to `models/object_detection/`.

## Run order

```bash
cd /work/GRDVULN/correction/scripts

# 0. Build training bins from the master gdb (rerun whenever the gdb changes)
sbatch slurm/00_build_training_bins.slurm

# 1a. Candidate parcels + NLCD, one array task per state
sbatch --array=0-1 --export=STATES="OH PA" slurm/01a_extract_parcels.slurm

# 1b. Object detection (NAIP streamed from Planetary Computer)
sbatch --array=0-1 --export=STATES="OH PA" slurm/01b_run_object_detection.slurm

# 2. Feature engineering -- ONE job across all states (comma-separated)
sbatch --export=STATES="OH,PA" slurm/02_feature_engineering.slurm

# 3/4. Train
sbatch slurm/03_train_stage1.slurm
sbatch slurm/04_train_stage2.slurm
```

Note the argument style difference: 01a/01b are **array** jobs and take
space-separated states (one per task); 02 is a **single** job and takes a
comma-separated list, because it builds one combined feature table.

## Resume

01b resumes by default. If a job hits its time limit or gets preempted,
resubmit the identical command — any `CWNS_ID` already written to
`data/od_features/plants/state=XX/` is skipped. It flushes every 200 plants, so
at most ~200 plants of work is lost on a kill. `NORESUME=1` forces a full
reprocess.

01a and 02 are not resumable; they're re-run from scratch.

## Files to place in scripts/

Copied unchanged from the local machine (config-driven, zero code changes needed):
- `01a_extract_parcels.py`
- `02_feature_engineering.py`
- `03_train_stage1.py`
- `04_train_stage2.py`
- `model_utils.py` (shared by 03/04 -- preprocessing, spatial CV folds, Youden threshold)

Rewritten for the HPC (Planetary Computer streaming instead of local .sid reads):
- `01b_run_object_detection.py`

New for this HPC port:
- `config.py`
- `build_training_bins.py`
- `setup_env.sh`, `_common.sh`, `slurm/*.slurm`

## Known gaps

- **Training-set size.** The local OH-only run produced 47 Correct / 1
  Incorrect for Stage 1 and 4 positives / 73,142 negatives for Stage 2.
  Neither is trainable. Check the class counts at the end of 02's log before
  submitting 03/04 — widening the state list is what fixes this.
- **NAIP worker count.** `NAIP_WORKERS = 32` is a starting point. Scaling
  tested roughly linear to 47 on 2026-08-20, but with only one trial per
  worker count and random sample points, so the ceiling isn't established.
  Watch 01b's fetch-failure count; a spike suggests rate-limiting.
- **Memory ceilings** in the slurm files are estimates from local runs
  (OH parcel attributes alone were 927k rows). Adjust after seeing real usage
  with `seff <jobid>`.
- **`K_RINGS = 18`** (~10km) is what produced 73k Stage 2 candidates per plant
  locally. It is carried over from the R pipeline as-is; if Stage 2 candidate
  volume becomes a problem on full-universe runs, that is the knob — but
  changing it changes what the model is trained on.
- **`Duplicate_Parcel_Flag`** in the master gdb is unreliable (computed
  without restricting to treatment-plant parcels) and is ignored everywhere.
  Worth recomputing correctly at some point.
