# correction/scripts — what runs when

**The numbers are IDs, not a running order.** They are stable labels the whole
codebase cross-references by (`see 01b's docstring`, `same pattern 01c uses`),
and they are worth keeping stable for that reason — but do not read them top
to bottom and expect a pipeline.

Three places they don't line up, all of them real:

- **`05` (inference) runs *after* `06`/`07`/`06b`/`07b` (training).** Inference
  loads trained models, so it cannot come first. This is the biggest one.
- **`01e` runs twice, at opposite ends of a cycle.** `SCOPE=train` early, to
  feed the re-ranker; `SCOPE=queue` at the very end, once the review queue
  exists and it knows which parcels to examine.
- **`09_build_holdout` runs once, ever.** It is setup, not a step.

Renumbering would not fix this. The pipeline is a DAG with a loop, and any
linearization of it lies somewhere — it would just move which part is wrong,
while breaking ~800 cross-references.

The authority on order is:

- **[docs/05_RUNBOOK.md](../../docs/05_RUNBOOK.md)** — the commands, in order
- **[docs/pipeline-map.html](../../docs/pipeline-map.html)** — Figure 2 is the
  real dependency graph; every argument of every script is below it

---

## Once, ever

| | |
|---|---|
| `setup_env.sh` | build the shared venv, on a **login node** |
| `09_build_holdout` | sample the frozen evaluation holdout and freeze it |

Re-running `09` after any training has happened invalidates every
round-over-round comparison the project has produced. Don't.

## Every cycle, in this order

| Order | Script | Note |
|---|---|---|
| 1 | `build_training_bins` / `00_build_training_bins.slurm` | normally runs locally inside `close_round.py` instead |
| 2 | `01a_extract_parcels` | array, **space**-separated states |
| 3 | `01b_run_object_detection` | detection at reported locations |
| 4 | `01c_run_od_corrected_locations` | detection at corrected locations |
| 5 | `01e_run_od_candidates.slurm` `SCOPE=train` | **feeds the re-ranker — skipping it is silent** |
| 6 | `check_od_freshness` | preflight; fails on detection output older than the deployed model |
| 7 | `02_feature_engineering` | array, one state per task, `--shard` |
| 8 | `02b_merge_feature_shards` | chain with `--dependency=afterok` |
| 9 | `03_train_stage1`, `04_train_stage2` | independent of each other |
| 10 | `06_build_stage2b_training` → `07_train_stage2b` | ordered pair |
| 11 | `06b_build_rerank_training` → `07b_train_rerank` | ordered pair, independent of 10 |
| 12 | `05_run_inference_array.slurm` + `merge_05_shards` | **after** training; the single job OOMs nationally |
| 13 | `05b_rerank_candidates` | needs `07b`'s model |
| 14 | `10_build_review_queue` | picks the plants |
| 15 | `01e_run_od_candidates.slurm` `SCOPE=queue,ROUND=N` | detection on the parcels that queue shows |
| 16 | `10_build_review_queue` again | same `--seed`; attaches the detection stats |
| 17 | `12_score_holdout` | the only honest read on whether any of it worked |

Then the round goes to the review app, and `close_round.py` brings it back.

**There are two 01e wrappers and they are not interchangeable.**
`01e_run_od_candidates.slurm` is the one that takes `SCOPE` — `train`,
`queue`, or unset for the holdout. `01e_run_od_candidates_array.slurm`
hardcodes `--all` and exists only to split the *national* run across states,
taking its state list as a positional argument rather than `--export`. Steps 5
and 15 above need the scoped one.

`01d_nlcd_topup` is a gap-filler, not a step: NLCD stats for a specific parcel
list that fell outside `01a`'s sweep.

## On demand

Diagnostics, run when a number looks wrong or **before** changing a modelling
decision — never as part of the sequence:

`08_diagnose_candidate_coverage`, `08b_analyze_ring_misses`,
`diagnose_candidate_od`, `diagnose_stage1_attrition`, `diagnose_od_class_shift`,
`inspect_model_features`, `list_training_states`, `check_review_queue_scores`,
`diagnose_01b_output`, `diagnose_fetch_failures`, `diagnose_parcel_duplicates`,
`diagnose_bytearray_columns`, `diagnose_scoreable_dtypes`,
`check_competition_losses`, `check_location_type`,
`check_location_type_overlap`, `inspect_stage2_columns`, `inspect_deployed_02`,
`check_01a_01b_complete`.

`collect_diagnostics` writes a text snapshot of this machine's pipeline state
to `correction/diagnostics/`, which is not gitignored — run it as a job,
commit the file, push. That is the working channel for getting cluster state to
a machine that cannot see the cluster. `check_parcel_coverage` is the narrower
check it wraps: which states have Regrid parcels at all, against the states
that have labelled plants.

`build_test_bundle` assembles a self-contained fixture so the pipeline can be
debugged in seconds instead of one `sbatch` per hypothesis. It is a fixture,
not a training set — a model trained on it must never be treated as real.

## Shared modules, not entry points

`config.py` — every path and tunable. `holdout.py` — `exclude_holdout()`,
fail-closed. `model_utils.py` — preprocessing, spatial CV folds, threshold
selection.

## Retired

**`11` is a retired slot, not a missing script.** `11_ingest_review_log.py`
derived training labels from review verdicts on the HPC, in parallel with the
master file doing the same thing locally. Deleted 2026-09-21: two paths
producing the same labels from the same verdicts is how they silently diverge.
Verdicts now reach training only through the master —
`review_app/sync/close_round.py`.

`patch_01e_training_scope.py` and `patch_05_for_array.py` are already applied.
They are a record of what changed, not something to run.

---

## The traps that fail silently

| | |
|---|---|
| Skipping `01e SCOPE=train` before `06b` | the round's hard negatives never reach the re-ranker; nothing errors |
| Running `02` as an array without `--shard` | 52 tasks overwrite one filename; last finisher wins |
| Skipping `02b` after the array | the flat tables stay as the previous run left them |
| Space vs comma separated `STATES` | `01a`/`01b`/`01c`/`01e` are arrays (space); `02` is single (comma) |
| Running `02` on stale detection output | mixes two models in one feature table — `check_od_freshness` exists for this |
| Re-running `09_build_holdout` | destroys every round-over-round comparison, undetectably |

Fuller list in [docs/06_TROUBLESHOOTING.md](../../docs/06_TROUBLESHOOTING.md).
