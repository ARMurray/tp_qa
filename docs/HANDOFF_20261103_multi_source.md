# Hand-off, 2026-11-03: multi-source seeds, HPC run in progress

Read this first, then [RUNBOOK_multi_source.md](RUNBOOK_multi_source.md) (commands, run log) and
[PLAN_20261103_multi_source_seeds.md](PLAN_20261103_multi_source_seeds.md) (design and decisions).

## Where things stand

- The multi-source pipeline is built, tested on synthetic data (`bash correction/tests/run_all.sh`), and on `master`
  (no branch, by decision). `TPQA_SEED_MODE=multi` is the default; outputs go to `*_multi` paths so the single-seed
  results are untouched.
- On the HPC (`/work/GRDVULN/tp_qa`), walked through one step at a time:
  - **01a** done, all 51 states (85.6M parcels, median state 1.09x the single-seed run).
  - **01f** done, all 51 states (55,945 seeds, 16,430 plants, 34,482 sites; 1.1% of seeds on no parcel).
  - **01e `--seed-sites`** queued the evening of 2026-11-03:
    `sbatch --array=0-47%6 --export=SEEDSITES=1 01e_run_od_candidates_array.slurm` (~17 h aggregate, rate-limited).
  - Everything from **02** on has not been run on real data.
- Progress is readable without HPC access: every job copies its log to `correction/diagnostics/latest_logs/`, refreshes
  `correction/diagnostics/multi_status.txt`, and pushes `correction/diagnostics/` to GitHub by itself. `git pull`, read the
  file. If a step's results are missing, the fallback is `bash correction/scripts/push_diagnostics.sh` on a login node.

## How the user works

The user runs commands on the HPC from a phone and cannot paste long output, so the scripts report through git. Give
them one copyable command per step, tell them what to check, and wait for them to say it is done. Never ask them to paste logs.
The HPC login shell is tcsh (`setenv`, not `export`).

## Decisions already made (do not reopen)

- Seeds = one plant-role location per source: `cwns_point`, `cwns_addr`, `icis_fac_frs`, `icis_fac_addr`, `icis_feat_INF`,
  `icis_feat_INO`. Default is to use all of them; `cwns_only` is a baseline only.
- Dedupe by parcel first: seeds on one parcel are one site and go through Stage 1 once. Adjacent parcels are merged only
  when the normalised owners match (within 50 m); a seed can sit on the wrong side of a line, so proximity alone never merges.
- If any site clears the plant-level Stage 1 threshold, the best one is taken (near-ties within 0.02 broken by number of
  independent origins, then by holding the CWNS point; a runner-up is kept). If none clears it, every parcel from the union
  of the seed windows goes to Stage 2, nothing excluded.
- The output records where each final location came from (`loc_basis`, `loc_sources`, `loc_n_origins`, `nearest_source`,
  `led_by`, `dist_to_final_<source>_m`); a combination of sources is expected and recorded.
- Reviewers did not use ICIS pins when finding the existing corrections, so the labels are not biased toward ICIS.

## Next steps

1. Read `multi_status.txt`, section "01e --seed-sites detection". Confirm every array task exited 0
   (`sacct -j <JOBID> -X --format=JobID%20,State,ExitCode`); resubmit the same command for anything that timed out (finished
   parcels are skipped). AK, HI and PR have no NAIP, so no detection there is expected.
2. `sbatch --export=FULL_UNIVERSE=1 02_feature_engineering.slurm`, then `02b_merge_feature_shards.slurm`. Check
   `Seed-site OD joined` (close to M outside AK/HI/PR), `Correct sites per plant` (mostly 0 or 1).
3. `03_train_stage1.slurm`. Read the "Plant-level accept analysis" table with the user and agree the precision target.
   03 refuses to train if a feature looks like the label (`find_label_leaks`); if it fires, treat it as a real bug.
4. `04_train_stage2.slurm`, then 05 inference array + `merge_05_shards.slurm`, the re-ranker steps, 05b, 12, 13 exactly as
   in the runbook "Order" section. Read `multi_status.txt` after each.
5. Compare END-TO-END right (12, multi) with the single-seed recall@1 on the same holdout (run 12 with
   `TPQA_SEED_MODE=cwns_only` for the old numbers).
6. Not started: review app showing each candidate's sources on its card.

## Known limits to watch

Stage 2 trains only on corrected plants, so it learns to rank the CWNS-point parcel low; watch the holdout `correct` bin
for plants wrongly moved. The plant-level threshold is calibrated on a noisy 25% hold-out. Simultaneous array finishes can
skip an auto-push (use the fallback script). Details in the runbook.

## Prompt for the new session

```
I'm continuing the tp_qa multi-source seeds work. The repo is C:\Github\tp_qa (local) and /work/GRDVULN/tp_qa (HPC), on
master. First read docs/HANDOFF_20261103_multi_source.md, docs/RUNBOOK_multi_source.md and
docs/PLAN_20261103_multi_source_seeds.md, then `git pull` and read correction/diagnostics/multi_status.txt.

Context: every plant-role location (CWNS point, CWNS geocode, ICIS-NPDES) now seeds the correction models, and the output
records which source(s) each final location came from. We are walking the HPC run one step at a time. 01a and 01f are done
for all states. I queued 01e --seed-sites (sbatch --array=0-47%6 --export=SEEDSITES=1 01e_run_od_candidates_array.slurm)
on the evening of 2026-11-03; I'm checking in the morning.

I'm usually on a phone: give me one copyable command per step, tell me what to check, don't ask me to paste logs (the jobs
push diagnostics to git; just pull and read). The HPC shell is tcsh.

Start by reading the 01e section of multi_status.txt and the latest_logs for 01e, tell me whether the detection array
finished cleanly (any failed or timed-out tasks, coverage of seed sites), and give me the command for the next step (02 then
02b). Then continue through 03/04/05/13/12 with me, one step at a time, reading results after each. Don't reopen settled
decisions (adjacent parcels merge only when owners match; dedupe by parcel before Stage 1; send all parcels to Stage 2
when no site passes). Flag anything in the results that looks wrong before moving on.
```
