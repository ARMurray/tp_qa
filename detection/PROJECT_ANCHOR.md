
/
Location Correction
Location Correction
A machine learning model that assesses the accuracy of treatment plant locations and corrects those that are incorrect








Recents
Treatment plant QA holdout validation and review loop setup
1 minute ago
Stage 2b training data strategy for wastewater treatment plant QA
7 hours ago
Running feature engineering on treatment plant detection output
yesterday
Location correction pipeline architecture and object detection integration
6 days ago
Integrating object detection into location correction workflow
Jul 21
Wastewater infrastructure spatial database crosswalk validation
Jul 9
Wastewater infrastructure database development and data cleanup
Jul 8
Integrating wastewater datasets for location data validation
Jul 7
Ohio sewershed and treatment plant visualization
Jun 9
Exporting model features for Shiny app review
May 27
Dynamic version selection for Quarto report
May 26
Building a Shiny app for model result review
May 26
Feature importance analysis for model retraining
May 21
Assessing permitted feature locational accuracy using water body data
May 20
Instructions
Add instructions to tailor Claude’s responses

Context
4% of project capacity used
Search mode

REVIEW_LOOP_PLAN.md
288 lines

md




SESSION_LOG_2026-08-24.md
174 lines

md




TPQA_PROJECT_STATUS_HPC.md
497 lines

md




OD_PROJECT_DOCUMENTATION.md
408 lines

md




app.R
1,380 lines

text




MODEL_NOTES.md
246 lines

md




FEATURE_ENGINEERING.md
282 lines

md




RESULTS_NOTES.md
273 lines

md




PROJECT_ANCHOR.md
297 lines

md




HPC_NOTES.md
372 lines

md




cwns_access_inventory.csv
csv




CWNSDatabaseDictionaryJanuary2025.xlsx
xlsx



Scheduled
Set up recurring tasks for this project.

PROJECT_ANCHOR.md


# CWNS Treatment Plant Location Correction — Project Anchor
 
## Project Goal
 
Automate the detection and correction of incorrect geographic coordinates for ~16,453 wastewater treatment plants in the EPA Clean Watersheds Needs Survey (CWNS) dataset. Visual inspection of a sample found ~40% of reported locations were on the wrong land parcel, with a median correction distance of 1.4km and some errors exceeding 100km.
 
## Overall Architecture
 
```
CWNS PHYSICAL_LOCATION.txt (16,453 plants)
         │
         ▼
┌─────────────────────────┐
│  STAGE 1                │
│  Score reported parcel  │
│  RF classifier          │
│  Threshold: 0.834       │
│  Output: prob_correct   │
└─────────────────────────┘
         │
    prob < 0.834 OR no parcel found?
         │
         ▼
┌─────────────────────────┐
│  STAGE 2                │
│  Search k=18 H3 rings   │
│  (~10km radius)         │
│  RF ranker              │
│  Threshold: 0.298       │
│  Output: ranked         │
│  candidate parcels      │
└─────────────────────────┘
         │
         ▼
    inference_results + geopackage
```
 
## Pipeline Scripts
 
Scripts are organized into numbered folders on the HPC reflecting pipeline progression order.
 
| Folder | Script | SLURM | Purpose |
|--------|--------|-------|---------|
| `01_data_prep/` | `extract_parcels.R` | `extract_parcels.slurm` | NLCD extraction — SLURM array, one job per state, K_RINGS=18 |
| `02_feature_engineering/` | `build_features_part1.R` | `build_features_part1.slurm` | Plant features (runs once) |
| `02_feature_engineering/` | `build_features_part2.R` | `build_features_part2.slurm` | Parcel features — SLURM array, one job per state |
| `02_feature_engineering/` | `build_features_part3.R` | `build_features_part3.slurm` | Consolidate state parquets, build training pairs, write DuckDB |
| `03_models/` | `train_stage1.R` | `train_stage1.slurm` | Stage 1 model training |
| `03_models/` | `train_stage2.R` | `train_stage2.slurm` | Stage 2 model training |
| `04_inference/` | `run_inference_stage1.R` | `run_inference_stage1.slurm` | Stage 1 scoring — single job |
| `04_inference/` | `run_inference_stage2.R` | `run_inference_stage2.slurm` | Stage 2 candidate scoring — SLURM array, one job per state |
| `04_inference/` | `run_inference_combine.R` | `run_inference_combine.slurm` | Merge state Stage 2 parquets, write DuckDB |
| `05_results/` | `build_results.R` | (interactive) | Builds geopackage and summary CSVs from inference outputs |
 
**Inference was split into three scripts** (Stage 1, Stage 2 array, Combine) because the monolithic `run_inference.R` took ~10.5 hours. Stage 2 scoring (plant-by-plant candidate ranking across 55M parcels) was the bottleneck. The array approach runs one state per job simultaneously, reducing wall time to ~30 minutes per state.
 
**Feature engineering split:** `build_features.R` was split into three parts due to OOM failure at the LBCS reclassification step when processing 110M parcels at 180GB memory limit. Part 2 runs as a SLURM array (one job per state, 180GB each) to avoid loading the full national dataset into a single process.
 
## Inference Submission Order
 
```
1. sbatch run_inference_stage1.slurm         # wait for completion
2. sbatch run_inference_stage2.slurm         # array — wait for all states
3. sbatch run_inference_combine.slurm        # after all array jobs finish
4. Run build_results.R interactively
```
 
## Training Data Update Workflow (local → HPC)
 
When treatment plant training data needs updating:
 
1. Run `Update_Training_Plants.R` locally (in `Location_Correction/` folder)
   - Reads `Inspection.gdb` and `Updates.gdb`
   - Writes updated `training_locations.gpkg` with `classes` and `corrections` layers
2. Upload updated `training_locations.gpkg` to `02_feature_engineering/` on HPC
3. Re-run `build_features_part3.R` (Parts 1 and 2 unchanged)
4. Re-run `train_stage1.R` and `train_stage2.R`
5. Re-run inference pipeline
**Validation check after Part 3:** Corrected parcels count in Step 13 log should equal the number of corrections with `How_Corrected == "Parcel"` in Updates.gdb. Any shortfall means corrected locations are not intersecting parcels — investigate before training.
 
**Note on 35 missing positive labels:** 35 of 302 corrections do not appear as positive labels in Stage 2 training. Investigation confirmed these are gross location errors where the corrected parcel H3 distance from the reported location exceeds k=18 rings (~10km). These are expected exclusions — Stage 2 cannot find parcels beyond its search radius at inference either.
 
## Current Status (as of May 2026)
 
### First inference run (k=9, original models) — COMPLETE
- [x] NLCD extraction (55 states, k=9 rings, ~65M parcels)
- [x] Feature engineering (`final_features.duckdb`, 65M parcels)
- [x] Stage 1 model trained (ROC AUC 0.986, threshold 0.677)
- [x] Stage 2 model trained (ROC AUC 0.975, threshold 0.057)
- [x] Inference run complete (16,453 plants processed)
- [x] Results downloaded locally
- [x] Quarto report — COMPLETE
### Second inference run (k=18, retrained models) — COMPLETE
- [x] NLCD extraction rerun (55 states, k=18 rings, ~110M parcels)
- [x] build_features_part1.R — plant features unchanged
- [x] build_features_part2.R — parcel features rebuilt at k=18
- [x] build_features_part3.R — training pairs rebuilt with updated training_locations.gpkg
- [x] Stage 1 retrained (ROC AUC 0.987, threshold 0.834)
- [x] Stage 2 retrained (ROC AUC 0.992, threshold 0.298)
- [x] Inference run complete (16,453 plants processed)
- [x] build_results.R run with post-processing additions
- [x] Results downloaded locally
### Pending for next run
- [ ] Add `subdivision`, `place`, `county` to `drop_cols` in both training scripts (fix already made — will take effect on next retrain)
- [ ] Add new manually reviewed training cases to Updates.gdb
- [ ] Retrain and rerun inference
## File Paths
 
### HPC (`/work/GRDVULN/`)
```
data/
  parcels/state=XX/*.parquet              # Regrid parcel data, hive partitioned
  nlcd/Annual_NLCD_LndCov_2023_CU_C1V1.tif
  Census/tlgdb_2022_a_us_substategeo.gdb
  osm/Wastewater_Plants.gpkg
 
Location_Repair/
  01_data_prep/
    extract_parcels.R                     # NLCD extraction, SLURM array by state
    extract_parcels.slurm
 
  02_feature_engineering/
    build_features_part1.R + .slurm       # Plant features
    build_features_part2.R + .slurm       # Parcel features, SLURM array by state
    build_features_part3.R + .slurm       # Consolidation, training pairs, DuckDB
    ready_check.R                         # QA helper
    CWNS_files/
      PHYSICAL_LOCATION.txt
      FACILITY_TYPES.txt
      DISCHARGES.csv
      POPULATION_WASTEWATER.txt
      FACILITIES.txt
      FACILITIES_CONFIRMED.txt
      FACILITY_PERMIT.csv
    training_locations.gpkg               # classes + corrections layers (updated locally)
    nlcd_outputs/
      nlcd_XX_k18.parquet                 # per-state NLCD extraction outputs (k=18)
    outputs/
      01_plant_base.parquet
      02_discharge_features.parquet
      03_population_features.parquet
      04_census_features.parquet
      05_plant_features.parquet           # combined plant features (16,453 rows)
      12_reported_parcels.parquet
      13_corrected_parcels.parquet
      14_stage1_training.parquet
      15_stage2_training.parquet
      final_features.duckdb               # tables: plant_features, parcel_features,
                                          #         stage1_results, stage2_results,
                                          #         inference_results
      parcel_attrs_by_state/
        parcel_attrs_XX.parquet           # one file per state, used by Stage 2 inference array
 
  03_models/
    train_stage1.R + .slurm
    train_stage2.R + .slurm
    stage1_rf_model.rds
    stage2_rf_model.rds
    stage1_optimal_threshold.rds          # 0.834 (second run)
    stage2_optimal_threshold.rds          # 0.298 (second run)
    stage1_rf_importance.parquet
    stage2_rf_importance.parquet
    stage1_cv_comparison.parquet
    stage2_cv_comparison.parquet
    stage1_threshold_analysis.parquet
    stage2_threshold_analysis.parquet
 
  04_inference/
    run_inference_stage1.R + .slurm       # Stage 1 scoring, single job
    run_inference_stage2.R + .slurm       # Stage 2 scoring, SLURM array by state
    run_inference_combine.R + .slurm      # Merge state parquets, write DuckDB
    stage1_results.parquet                # 16,453 rows, one per plant
    stage2_results.parquet                # 391,277 candidates (second run, post-exclusion)
    final_results.parquet                 # stage1_results + reported_ll_uuid
    stage2_by_state/
      stage2_plants_XX.csv               # per-state Stage 2 plant lists (Stage 1 output)
      stage2_results_XX.parquet          # per-state Stage 2 results (Stage 2 array output)
 
  05_results/
    build_results.R                       # interactive, run after inference
    results_download/
      parquet/
        stage1_results.parquet
        stage2_results.parquet
      plant_summary.csv
      inference_results.gpkg
      summary_overall.csv
      summary_stage1_score_distribution.csv
      summary_top_s2_score_distribution.csv
      summary_tier_summary.csv
 
  logs/                                   # all SLURM logs consolidated here
```
 
### Local (repo: `C:/Users/AMURRA02/.../Github/`)
```
Location_Correction/
  training_locations.gpkg                 # master training data — updated here, uploaded to HPC
  Update_Training_Plants.R               # script to rebuild training_locations.gpkg
  data/Updates.gdb                       # master corrections file
 
Sewersheds/
  Location_Correction/
    results_download/                    # downloaded from HPC after each run
    models/
      stage1_rf_model.rds
      stage2_rf_model.rds
 
  Data/
    PHYSICAL_LOCATION.txt
    FACILITIES.txt / FACILITIES_CONFIRMED.txt
    FACILITY_PERMIT.csv
    FACILITY_TYPES.txt
    DISCHARGES.csv
    POPULATION_WASTEWATER.txt
 
  Analysis/Location_Matching/Data/Inspection.gdb
  Data/OSM/Wastewater_Plants.gpkg
  Data/Regrid/Parquet_Storage/
  Data/MRLC/Annual_NLCD_LndCov_2023_CU_C1V1.tif
  Data/Census/tlgdb_2022_a_us_substategeo.gdb
```
 
## Data Sources
 
| Source | Description | Format | Key Columns |
|--------|-------------|--------|-------------|
| CWNS PHYSICAL_LOCATION | Treatment plant coordinates | CSV | CWNS_ID, LATITUDE, LONGITUDE, STATE_CODE |
| CWNS FACILITY_TYPES | Facility type filter | CSV | CWNS_ID, FACILITY_TYPE |
| CWNS DISCHARGES | Discharge type (replaces NPDES join) | CSV | CWNS_ID, DISCHARGE_TYPE, PRESENT_DISCHARGE_PERCENTAGE |
| CWNS POPULATION_WASTEWATER | Population served | CSV | CWNS_ID, TOTAL_RES_POPULATION_2022 |
| Regrid | Land parcel data | Parquet (hive by state) | ll_uuid, h3_index_9, wkb_geometry, lbcs_*, owner, geoid |
| NLCD 2023 | Landcover raster | GeoTIFF (Albers, 30m) | pixel values (class 11 = open water) |
| OSM | Wastewater facility locations | GeoPackage | Points layer, geom column |
| Census 2022 | Geographic boundaries | FGDB | County_Subdivision, Incorporated_Place, County |
| Updates.gdb | Manual location corrections | FGDB | CWNS_ID, Original_X/Y, Corrected_X/Y, How_Corrected |
 
## Spatial Infrastructure
 
- **H3 resolution 9**: ~0.1 km² per hexagon, ~174m edge length, ~330m per ring step
- **k=18 rings**: ~1,009 hexagons, ~10km search radius for Stage 2 (increased from k=9)
- **k=1 expansion**: Used for point-in-parcel lookups to handle H3 boundary cases
- **CRS**: Parcels stored in WGS84 (EPSG:4326); NLCD in Albers (EPSG:5070); census joins in NAD83/Conus Albers (EPSG:5070)
- **Coverage at k=18**: ~110M parcels (~65M at k=9) — effectively near-national coverage given density of treatment plants
## Results Tier Definitions
 
| Tier | Condition | Action |
|------|-----------|--------|
| Tier1_nearby_correction | low_confidence AND candidate within 5km | High priority review |
| Tier1_distant_correction | low_confidence AND candidate beyond 5km | Review with caution |
| Tier1_no_candidate | low_confidence AND no Stage 2 candidate above threshold | Flag for manual review |
| Tier2_no_parcel_corrected | no parcel intersection AND candidate found | Definite error, strong correction |
| Tier2_no_parcel_uncorrected | no parcel intersection AND no candidate | Flag for manual review |
| Tier3_likely_correct | Stage 1 prob >= 0.834 | Likely correct, spot check only |
 
## Stage 2 Trigger Logic
 
```r
run_stage2 <- (stage1_prob_correct < 0.834) | (no_parcel_found == TRUE)
```
 
Plants with no parcel are definitionally wrong — the point doesn't land on any land parcel (in water, road, data gap, or genuinely incorrect).
 
## build_results.R Post-Processing (implemented)
 
Both of the following are implemented in `build_results.R` as of the second inference run:
 
1. **Exclude reported parcels from Stage 2 candidates** — Before ranking, all `reported_ll_uuid` values from `final_results.parquet` are removed from `s2`. Prevents self-correction and cross-plant contamination. Exclusion is unconditional. Applied before tier assignment.
2. **Resolve Stage 2 candidate competition** — When multiple Stage 2 plants share a candidate parcel, the parcel is assigned to the highest-scoring plant only at the `top_s2` selection step. The full `s2` table is kept intact so competing assignments remain visible for QA. Competition resolution only affects final tier assignment, not the candidate pool.
## Known Issues and Limitations
 
1. **Training parcel match rate**: In the original run, only 619 of 1,405 labeled plants (44%) intersected a parcel during Stage 1 training — likely due to rural data gaps and H3 boundary issues. Fixed in inference with k=1 expansion. Training data now updated with 2,746 labeled plants (2,348 correct, 398 incorrect).
2. **Stage 2 positive labels**: 168 of 302 corrections become positive labels. 35 are excluded because their corrected parcel H3 distance exceeds k=18 (gross errors). Remaining shortfall is parcel coverage gaps.
3. **No parcel states**: AS (American Samoa) has no Regrid parcel coverage.
4. **Novel level warnings at inference**: Census geography raw columns (`subdivision`, `place`, `county`) were included in training as factors, generating ~1,037 geography dummies. At inference nearly all levels are novel, coerced to NA by `step_novel()`. Fix implemented: these columns are now in `drop_cols` in both training scripts and `prep_features()` in inference scripts. Will take effect on next retrain.
5. **log_gisacre dominance in Stage 2**: Parcel size is a top Stage 2 feature and harms scoring of small correct parcels. `has_ww_keyword` partially mitigates this.
6. **OSM mislocation**: `osm_ww` can be tagged to an adjacent park or recreation parcel rather than the actual treatment plant. Confirmed in Baden Borough PA case.
7. **Stage 2 scoring runtime**: ~10.5 hours for full national run as single job. Addressed by splitting into SLURM array by state.
## Future Work (Deferred)
 
- Weekly scheduling pipeline (trigger script generates input CSV from internal DB changes)
- XGBoost comparison (blocked by Intel oneAPI compiler issue on HPC — `-O3` flag conflict)
- Owner name classifier from TF-IDF analysis
- Census population ratio features (`pct_county_pop_served`)
- Spatial leakage investigation in cross-validation
- Retrain Stage 1 on full 16k plant universe (currently trained on subset that intersected parcels)
- Geocoding pre-processing for gross location errors (>10km from address) — deferred in favor of k=18 expansion
- Prune low-importance features from training (negative permutation importance features identified — see MODEL_NOTES.md)
 
