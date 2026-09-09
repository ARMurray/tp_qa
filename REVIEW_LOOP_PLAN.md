# Review Loop Implementation Plan

**Status:** proposed, 2026-08-24
**Goal:** a repeatable cycle — inference → sampled review → retrain — that grows
Stage 2b's training set from 428 rows toward several thousand, while producing an
uncontaminated measure of whether each round actually improved anything.

---

## 0. The two rules that make this work

**Rule 1 — The holdout is sampled ONCE, never resampled.**
It is drawn from the plant universe before the first inference run and frozen as a
manifest. Every subsequent round reads that manifest. Plants in it never enter
training, in any round, in any form. Resampling per round would rotate plants
between train and eval and silently destroy every comparison.

**Rule 2 — The holdout is reviewed once, then scored automatically forever.**
A holdout with no labels is useless. Review each holdout plant once to establish
its true parcel (`ll_uuid` where one exists, coordinates otherwise). From then on,
scoring any model version is a join, not a review task: does the new top-1 candidate
match the stored truth?

---

## Phase 1 — Freeze the holdout (before any full inference run)

**New script: `09_build_holdout.py`**

- Draw a stratified random sample of plants, target ~250–300.
  Stratify by: state, and by whether the plant is in the existing training set.
- Include plants from at least one state absent from training, to expose domain
  shift in Regrid attribute completeness and LBCS coverage.
- Write `data/holdout/holdout_manifest.parquet`:

  | column | meaning |
  |---|---|
  | `CWNS_ID` | plant |
  | `STATE_CODE` | |
  | `stratum` | sampling cell |
  | `in_original_training` | was this plant in the 2026-08 training set |
  | `sampled_at` | timestamp |
  | `sample_seed` | RNG seed used |

- **Idempotence guard:** the script refuses to overwrite an existing manifest
  without `--force-resample`, and `--force-resample` prints a loud warning that
  all historical eval numbers become incomparable. This guard is the enforcement
  mechanism for Rule 1; without it the rule is just a comment.

Plants in the manifest are excluded from training in `03`, `04`, `06`, and `07`
by an anti-join on `CWNS_ID`. Add that anti-join in the same commit as the
manifest — a manifest nothing reads is worse than no manifest, because it looks
like protection.

**Second output: `data/holdout/holdout_truth.parquet`** — empty at first, filled
by the round-1 review pass. Columns: `CWNS_ID`, `true_ll_uuid`, `true_lon`,
`true_lat`, `truth_source` (`reported_correct` | `candidate` | `manual`),
`reviewed_at`, `reviewer`.

---

## Phase 2 — Pilot inference (2–3 states, not national)

Run the full pipeline end-to-end on a small state set including one out-of-training
state. Purpose is to shake out schema and tooling problems before generating
15,000 rows in a format the app can't read. Expected discoveries: column-name
drift between the Python outputs and what `app.R` expects, OD coverage gaps in the
new state, parcel-attribute sparsity.

Review ~100 plants through the app. Fix what breaks. Only then scale.

---

## Phase 3 — Build the review queue

**New script: `10_build_review_queue.py`**

Reads inference output, writes `Version_N/review_queue.parquet` — the thing the
app loads.

Composition, per batch of ~150 plants:

| slice | share | selection rule | purpose |
|---|---|---|---|
| `holdout` | round 1 only | from manifest, unreviewed | establishes the benchmark |
| `uncertain` | ~70% | small margin between top-1 and top-2 Stage 2b score; or Stage 1 says wrong but best Stage 2a candidate scores low; or no candidate above threshold | maximum information per minute |
| `random` | ~30% | uniform over all plants regardless of score | unbiased error rate; the only way to find confident-and-wrong failures |

The random slice is not optional and not droppable when it feels unproductive.
It is the calibration, and it is the only slice that can discover a blind spot
the model does not know it has.

Queue schema adds to each candidate row: `queue_slice`, `model_version`,
`candidate_rank`, `stage2b_score`, `score_margin`.

---

## Phase 4 — Extend `app.R` (four changes, not a rewrite)

The existing app already provides: versioned folders, leaflet + parcel polygons,
candidate table, candidate-level `review_log.csv`, and the
correct / incorrect / needs_info / reported_correct verdict paths.

Needed additions:

1. **"Truth is not in this list"** — a fourth submission path that opens a
   map-click or coordinate-entry control to capture the real location. Today the
   app can only say "reported location correct" or pick from the candidates, so
   the most valuable case — the model's pool missed entirely — has nowhere to go.
   Writes `plant_verdict = "truth_outside_candidates"` plus the captured point.

2. **Rank capture** — persist `candidate_rank` of the selected candidate, and
   `truth_rank = NA` for the case above. This single field separates the two
   failure modes: truth present but mis-ranked (Stage 2b's problem) vs. truth
   absent from the pool (candidate generation / `K_RINGS`). Cannot be
   reconstructed after the fact.

3. **Holdout routing** — the app reads the manifest and, on submit, routes
   holdout verdicts to `holdout_truth.parquet` instead of the training feed.
   Show a badge in the UI so the reviewer knows. After round 1, holdout plants
   are filtered out of the queue entirely.

4. **Provenance fields** on every logged row: `model_version`, `queue_slice`,
   `review_round`, `reviewed_at`, `reviewer`, and `confirmation_type`
   (`independent` = reviewer found it unaided, vs. `confirmed_proposal` =
   reviewer agreed with the model's top pick). The last one matters because
   confirming a model's answer is weaker evidence than finding the answer
   independently — flag it now, decide whether to weight it later.

---

## Phase 5 — Fold back into training

**New script: `11_ingest_review_log.py`**

- Reads `Version_N/review_log.csv`, anti-joins the holdout manifest (belt and
  braces — the app should already have routed these away).
- Emits distribution-matched Stage 2b rows: for each reviewed plant, the selected
  candidate is `label=1` and every other candidate in the ranked list is
  `label=0`. One K=5 review yields 1 positive and 4 hard negatives drawn from the
  actual inference distribution — which is precisely what `06`'s reported-parcel
  negatives are not.
- `truth_outside_candidates` plants emit negatives only, plus a row in a
  `candidate_recall_failures` table feeding the `K_RINGS` decision.
- Appends to a cumulative `data/features/17_review_training.parquet` with all
  provenance columns retained. `06` unions this with its existing output.

**Retrain in batches of ~100–200 verdicts, not continuously.** At n=428, twenty
new rows move metrics by noise.

---

## Phase 6 — Score and log

**New script: `12_score_holdout.py`** — run after every retrain.

Joins the new model's top-1 candidate per holdout plant against
`holdout_truth.parquet`. Reports:

- `recall@1`, `recall@3`, `MRR`
- `candidate_recall` — fraction of holdout plants whose true parcel appears
  anywhere in the pool. This is the pipeline ceiling; `recall@1` cannot exceed it,
  and if the gap between them is small, ranking is not the bottleneck.
- Breakdown by `in_original_training` stratum and by state.

Appends one row per model version to `RESULTS_NOTES.md` and to
`data/holdout/holdout_scores.parquet`, so the curve accumulates rather than
being a series of disconnected numbers.

**Exit criterion — decide the number before round 1, not after seeing results:**
loop continues until `recall@1` on the frozen holdout reaches the agreed target,
or until two consecutive rounds produce no improvement outside noise (judge
against the spread across random seeds, given the sample size).

---

## Build order

1. `09_build_holdout.py` + the anti-joins in `03`/`04`/`06`/`07`
2. `app.R` changes 1 and 2 (truth-outside-candidates, rank capture)
3. Pilot inference on 2–3 states
4. Round-1 review: holdout first, to establish the benchmark
5. `10_build_review_queue.py`, `11_ingest_review_log.py`, `12_score_holdout.py`
6. First full cycle

Steps 1–2 gate everything else. Reviewing before rank capture exists means
re-reviewing later to recover it.

---

## Open decisions

- **Holdout size.** 250–300 plants gives a recall@1 standard error around ±3pp.
  Tighter needs a bigger holdout, which costs training data you can't spare.
- **K in the candidate list shown to the reviewer.** Larger K means more hard
  negatives per review and a better `candidate_recall` estimate, but slower
  reviews. K=5 is a reasonable start; revisit once per-review timing is known.
- **Weighting of `confirmed_proposal` rows** relative to `independent` ones.
  Defer, but capture the field now.
- **Whether `needs_info` plants re-enter the queue** in later rounds once the
  model changes, or are permanently set aside.

---

## Phase 7 — HPC / local split and round folders

**HPC** (`/work/GRDVULN/correction/`) runs Phases 1–3 and 6: holdout sampling,
inference, queue construction, scoring. **Local** (`tp_qa/review/`) runs Phase 4–5:
the app and the ingest script.

### Round folder, built on HPC and downloaded whole

```
tp_qa/review/
  app.R
  ingest_review_log.R          # local, R -- see below
  Round_03/
    round_manifest.json        # written by HPC, verified by local
    stage1_results.parquet     # names kept from app.R's existing convention
    stage2_results.parquet
    plant_summary.csv
    review_queue.parquet       # from 10_build_review_queue.py
    holdout_manifest.parquet   # read-only copy, checksum-verified
    review_log.csv             # written by app.R in place
    holdout_truth_round03.parquet   # round 1 only, in practice
    17_review_training_round03.parquet  # written by ingest, uploaded back
```

### Sync contract

**Down (HPC → local):** everything except `review_log.csv` and the two files the
ingest writes.

**Up (local → HPC):** `review_log.csv`, `holdout_truth_roundNN.parquet`, and
`17_review_training_roundNN.parquet` only. Uploads land in a round-scoped path
and are never overwritten.

### Four rules for the split

1. **Deltas up, never cumulative.** Local writes only that round's new training
   rows. `06` on HPC globs `17_review_training_round*.parquet` and concatenates.
   A stale or partial local cumulative file would otherwise erase earlier rounds'
   labels with no error raised.

2. **Holdout manifest is read-only off-HPC.** `ingest_review_log.R` verifies its
   checksum against `round_manifest.json` and stops on mismatch. It never
   regenerates or repairs it. This is the local-side enforcement of Rule 1.

3. **Ingest is written in R.** It lives beside `app.R`, which already loads
   `arrow`, `tidyverse`, and `sf`. A Python ingest would require maintaining a
   second local environment for what is a reshape from candidate-level verdicts
   to labeled training rows. HPC stays Python; the interface is parquet.

4. **`round_manifest.json` in every round folder.** Written by HPC, checked by
   local on load. Contents: round number; Stage 1 / 2a / 2b model file hashes;
   inference date; states covered; `K_RINGS`; holdout manifest checksum; queue
   row and plant counts; slice proportions actually achieved.

   Without it, the `model_version` provenance field recorded in Phase 4 has
   nothing authoritative to resolve against, and several rounds in there is no
   reliable way to say which model produced the candidates a given verdict judged.

### `ingest_review_log.R` responsibilities

- Verify `round_manifest.json` and the holdout checksum; stop on mismatch.
- Split `review_log.csv`: holdout plants → `holdout_truth_roundNN.parquet`,
  everything else → training rows.
- Expand each reviewed plant into one `label=1` row for the selected candidate
  and `label=0` for every other candidate in its ranked list.
- `truth_outside_candidates` plants → negatives only, plus a row in the
  candidate-recall failure table.
- Carry all Phase 4 provenance fields through unchanged.
- Refuse to run twice against the same round without `--force` (double-ingest
  would double-count every label).

### Retention

Round folders are kept indefinitely (decided 2026-08-24 — disk is not a
constraint). They are the audit trail: re-opening an earlier round's exact
candidate list is the only way to re-examine a verdict that later looks wrong,
and `stage2_results.parquet` cannot be regenerated once the model that produced
it has been retrained.

Treat completed round folders as immutable. If a verdict needs revising, correct
it in the current round's log with a note, rather than editing a past
`review_log.csv` — otherwise the provenance fields point at a file whose contents
have since changed.
