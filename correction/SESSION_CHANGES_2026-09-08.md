# SESSION_CHANGES_2026-09-08.md

Covers 2026-09-03 to 2026-09-08. Supersedes parts of
`TPQA_MASTER_REFERENCE.md` and `DETECTION_PIPELINE_REFERENCE.md` — the
corrections are listed in §1, read those first.

Three things happened: the detection labeling loop was repaired and a new
object-detection model trained; the correction pipeline was re-run at
national scale on those weights; and a **Stage 2a re-ranker** was built,
which overturns this project's standing conclusion that object detection
contributes nothing.

---

## 1. Corrections to existing documentation

**These prior conclusions are now known to be wrong. Do not act on them.**

### 1.1 "OD features contribute almost nothing" — WRONG, and consequentially so

`TPQA_MASTER_REFERENCE.md` §3/§10 and `05_run_inference.py`'s docstring both
assert this, citing Stage 2b importances near zero. That measurement was
real but was answering the wrong question.

`06_build_stage2b_training.py` trains Stage 2b on the **reported** parcel
(label 0) against the **corrected** parcel (label 1). Those two are usually
nothing alike — a residential lot versus a 40-acre municipal parcel on a
river. Parcel attributes separate them trivially, leaving OD no residual
variance to explain. Its importance collapsing was a fact about the training
pairs, not about object detection.

Measured directly, with no model in the loop, among Stage 2a's top-20
candidates (`diagnose_candidate_od.py`, holdout; `06b` dry run, training):

| | true parcel | competitors | ratio |
|---|---|---|---|
| detection rate (holdout) | 46.4% | 9.0% | 5.2× |
| detection rate (training) | 54.8% | 7.5% | 7.3× |
| `od_n_objects` (holdout) | 2.64 | 0.32 | 8.3× |

Ranking by OD confidence alone, no model, picked the true parcel first
35.7% of the time against a 5% random baseline.

**Do not re-derive "OD doesn't matter" from Stage 2b's importances again.**
If that conclusion is needed, measure it on candidate-distribution data.

### 1.2 `03_prepare_dataset.py` silently dropped every folded 500m tile

`fold_in_500m.py`'s docstring claimed 03 "picks them up automatically." It
never did. `parse_tile_stem`'s regex required an `r##_c##` suffix, and
`convert_tiles_to_500m.py` names its output `{cwns}_{uuid}_500m`. All 65
folded tiles failed to parse and appeared only in a "Could not parse"
warning. The same regex also rejected `TRI_{id}_r01_c01`. Fixed 2026-09-03.

### 1.3 `label_app.R` §4 is stale

The reference doc says "no in-place class change or box resize — only
delete-and-redraw." The app was rewritten 2026-09-02 to a canvas editor with
drag-to-move and resize. §4 predates that.

### 1.4 `K_RINGS` comment

`config.py` says k=18 is "~10km". `08_diagnose_candidate_coverage.py`
measures the effective search **radius** at ~5.6 km (311 m median per ring
step). The comment is describing the diameter.

---

## 2. Detection pipeline (local, `detection/`)

### 2.1 Label inventory reconciled

330 label `.txt` files against 65 tiles in `rgb_500/` looked like data loss.
It was not. `diagnose_label_inventory.py` partitioned them:

```
legacy_mangled   262 files   512 boxes
clean_500m        65 files   392 boxes
clean_200m         3 files    10 boxes
```

249 old files were superseded by a 500m tile; 16 were not. Those 16 belong
to the **6 of 71 groups** whose 500m footprint straddles a NAIP quad
boundary — the documented ~8.5% unrecoverable rate, independently
rederived from filenames rather than from the converter's run log. Their
PNGs are quarantined at `data/tiles/quarantine_200m/`.

### 2.2 Tiles moved to 500m natively

`config.py`: `TILE_SIZE_M` 200 → 500, `OVERLAP_PCT` 0.33 → 0. `IMAGE_PX`
follows to 833 and `MODEL_IMGSZ` to 864 automatically.

**This resolves the mixed-tile-size concern in the reference doc's §7.**
With the 200m tiles archived, every live tile is 833px and nothing is being
rescaled against anything else.

### 2.3 `01_sample_sites.py` / `02_extract_tiles.py` rewritten

- `--source {plants,review,tri,all}` and `--n` / `--n-tri`
- New `review_fp` layer sourced from `review_app/analysis/find_false_top_picks.py`
  — parcels the model ranked **above** the correct answer, i.e. confirmed
  visual false positives. The "visually confusable" negative category that
  script's own docstring notes was missing.
- Imagery: dead USDA ArcGIS → Planetary Computer, via a new shared
  `naip_fetch.py` that `02` and `convert_tiles_to_500m.py` both import
  rather than each carrying a copy of the retry/quad-fallback logic.
  Thread-local rasterio caches restored (the convert script's plain dicts
  are safe only because it runs sequentially; `02` uses a ThreadPoolExecutor).
- **Tiling: one-or-nine.** Every site gets a tile on its parcel's true
  centroid; if the parcel overruns that footprint, the ring of 8 is added.
  Never more. Rows are always numbered as a 3×3, so the centroid tile is
  `r02_c02` whether or not the ring exists.
- `--dry-run` reports the 1-vs-9 split and total tile count before fetching.
- Fixed in both: `sys.path.insert(0, parents[1])` put `detection/` ahead of
  `pipeline/`, so a `config.py` at the repo root would silently have won.

`rank1_ll_uuid` is the authoritative locator for review sites, **not**
`rank1_lat/lon` — for polygons those come from `_flatten_coords()`, a vertex
mean the source script itself flags as "not a real centroid," which on an
L-shaped parcel can fall outside the parcel. Rows whose uuid can't be
resolved are skipped rather than tiled from the approximate point.

### 2.4 New detection model

15 review sites → 63 tiles (4.2 tiles/site; 40% of parcels overran 500m).
Dataset: 128 tiles, 418 boxes. Filtered to 3 classes
(`aeration_basin`, `clarifier`, `digester`); `chlorine_contact`,
`drying_bed`, `oxidation_pond` dropped on volume — 6 training boxes for
drying_bed is not learnable.

**`oxidation_pond`'s exclusion comment is now stale.** It says they "need a
larger tile size than this model's 200m tiles support." At 500m that
objection is gone; the class is out on volume alone (36 train / 2 val) and
is worth revisiting after another labeling round.

Final validation (`wwtp_v2-4`, imgsz 864, 3 classes):

```
mAP50 0.662   mAP50-95 0.321   P 0.607   R 0.696
  aeration_basin 0.488 (12 val instances)
  clarifier      0.791 (36)
  digester       0.706 (11)
```

**Open question:** mAP50-95 is 48% of mAP50 — boxes are found but loosely
localized. Oddly, `clarifier` has the best AP50 and the *worst* mAP50-95.
Circular objects should box tightly. One candidate explanation is the
bounding-box drift observed at some plants during review: offset ground
truth clears IoU 0.5 but collapses at 0.75+. Unresolved.

### 2.5 GPU

`torch 2.13.0+cpu` had no CUDA at all. The card is an **RTX PRO 2000
Blackwell, sm_120** — needs cu128 or newer specifically; cu124/cu126 wheels
install cleanly, report `is_available() == True`, then fail at the first
real op with "no kernel image available." Resolved to `torch 2.11.0+cu128`.
`BATCH_SIZE` 16 → 4 (imgsz 352 → 864 is ~6× the activation memory).

---

## 3. Correction pipeline (HPC) — national run

Full sequence, all on the new weights:

```
01a → 01b → 01c → 01d → 02 → 03 → 04 → 06 → 07 → 05 → 05b
```

### 3.1 Invocation conventions differ per script — this is a real trap

| script | states passed as |
|---|---|
| `01a`, `01b`, `01d` | `--export=STATES="AL AR ..."` space-separated |
| `02`, `05`, arrays | **positional**, quoted |
| `05_run_inference.py` | `--states OH,MS,DE` comma-separated |
| `01c` | no states at all |

`sbatch --export` splits on commas itself, so a comma-separated list passed
that way is silently truncated at the first element. Confirmed 2026-08-26
(`--export=STATES="OH,MS,DE"` ran as just `OH`) and preserved in `02`'s and
`05`'s headers.

### 3.2 Resume semantics — the other real trap

| script | resume granularity | after an OD model change |
|---|---|---|
| `01a` | whole state | `NORESUME=1` + `--full-universe`, or it silently no-ops |
| `01b`, `01c`, `01e` | per plant / per (plant, parcel) | **`NORESUME=1` required** |
| `02` onward | none — full rebuild | n/a |

**`01b`/`01c`/`01e` resume cannot tell which model wrote a row.** Leaving
resume on after swapping weights preserves old-model output for every
already-processed plant, producing a table that is a silent mix of two
models. `check_od_freshness.py` now guards this: it compares every
partition's parquet mtime against the deployed `best.pt` and exits 1 on
anything older. Chain it ahead of `02`:

```
python $SCRIPTS/check_od_freshness.py --allow AK HI PR || exit 1
```

**Known gap:** the check compares the *newest* parquet per partition. `02`'s
run found 4,334 duplicate CWNS_IDs — old-model part files sitting *inside*
partitions that also contained fresh ones. `02` resolved them correctly by
mtime-recency, but the guard did not catch them. It should compare every
part file, not the newest per partition.

`01a` with `NORESUME=1` **overwrites the per-state parquets wholesale**,
discarding `01d`'s appended top-up rows. Order is always `01a` → … → `01d`.

### 3.3 Scale and results

- `01a`: 48 states, longest 30 min against a 12h limit
- `01b`: 48 states at `%6`, all COMPLETED, longest 30 min
- `01c`: 399/403 corrections, 6.7 min, 41/440 fetch failures (~9%, in band)
- `02`: 77,686,909 parcel rows, 6.5 hours at 200G
- `05`: OOM at 64G nationally — Stage 2a accumulates every state's
  candidates before scoring. Split into a per-state array (`--out-dir`
  patch + `merge_05_shards.py`); TX alone needed 200G. Result: 16,198
  plants, 240,546 candidates.

Model metrics:

```
Stage 1   test AUC 0.970   spatial 0.944 / standard 0.968
Stage 2a  test AUC 0.992   spatial 0.984 / standard 0.990
Stage 2b  test AUC 0.952   spatial 0.906 / standard 0.936
```

Stage 1 is dominated by `osm_ww` (0.300 importance; second feature 0.017).
It is close to an OpenStreetMap lookup with a small correction term.

### 3.4 Coverage gaps

- **AK, HI, PR** cannot be processed: NAIP is CONUS-only and
  `PROJECTED_CRS = 5070` is CONUS Albers. Their old-model partitions are
  quarantined at `data/od_features_stale/`.
- **DC** was omitted from the state lists by choice. `01c` processed 1 DC
  correction, so the corrected side covers a state the reported side does not.
- 23% of plants have no reported parcel at all (3,725 of 16,198).
- `08`: 91.1% of corrections have their true parcel inside the k=18 ring.
  The 8.9% that don't would need **k=238** for 95% coverage — ~175× the
  candidate pool. Not a viable fix; treat as a hard ceiling.

---

## 4. The re-ranker (new)

### 4.1 Why it exists

Stage 2a ranks 4.3M candidates on parcel attributes alone — object
detection on that many parcels is impossible. Stage 2b uses OD but was
excluded from inference. The re-ranker operates on the surviving **top-20**,
where OD is affordable and, per §1.1, actually discriminates.

### 4.2 New scripts

| script | does |
|---|---|
| `01e_run_od_candidates.py` | OD on top-K candidate parcels. Reuses `01b`'s `prepare_plant`/`run_batch`/`finalize_plant` via a composite `CWNS_ID::ll_uuid` key. Scopes: holdout (default), `--training-corrections`, `--all`. |
| `01e_run_od_candidates_array.slurm` | per-state array for the national run |
| `06b_build_rerank_training.py` | labels the true parcel 1, its ~19 competitors 0 |
| `07b_train_rerank.py` | trains it; reports **recall@k**, not AUC |
| `05b_rerank_candidates.py` | applies it, writing `rerank_score` alongside `stage2_prob_correct` |
| `diagnose_candidate_od.py` | measures true-vs-competitor detection, no model |

**One behavioural difference from `01b` worth knowing:** the NAIP item is
resolved from the **candidate parcel's own centroid**, not the plant's
reported point. A candidate can sit kilometres away — that is the point of
Stage 2a — and resolving from the reported point fetches the wrong quad and
fails every tile.

### 4.3 Results — four paired seeds, identical splits

| seed | Stage 2a | re-rank no-OD | re-rank +OD | OD gain |
|---|---|---|---|---|
| 42 | 54.8% | 67.7% | 75.8% | +8.1 |
| 7 | 56.5% | 66.1% | 71.0% | +4.9 |
| 13 | 66.7% | 71.4% | 76.2% | +4.8 |
| 99 | 59.7% | 65.3% | 73.6% | +8.3 |
| **mean** | **59.4%** | **67.6%** | **74.2%** | **+6.5** |

Spatial AUC, every seed: **0.965–0.970 with OD vs 0.950–0.952 without.**

A 4/4 sign test alone is p=0.06 — not conventionally significant. But these
are *paired* comparisons on identical splits, so the relevant noise is far
below the ±6pp marginal error, and two independent metrics agree in every
run. Treat the effect as real and the magnitude as approximate.

**Decomposition: ~60% of the gain (+8.2pp) needs no imagery at all.**
`--no-od` is deployable immediately and is worth most of the practical
value.

### 4.4 Caveats

- **recall@3 and @5 are at ceiling** (92–98%) and show no consistent OD
  benefit. All the OD value is in getting the right parcel to **position 1**.
  If reviewers see five candidates anyway, `--no-od` captures nearly
  everything. OD matters if you ever want to auto-accept a top pick.
- **Individual OD feature rankings are unstable** across seeds —
  `od_max_confidence` is 4th at seed 13 and absent from the top 20 at seed
  99. The *family* contributes; which member gets credit is noise at 263
  positives.
- **Candidate recall@20 differs between pools:** 81% training, 64% holdout.
  Small-sample noise explains some of it. If the holdout is genuinely
  harder, evaluation will read lower than training suggests.
- **The binding constraint has moved.** With recall@5 at ~97%, ranking is
  close to solved. What remains is candidate generation: 19 of 81 test
  plants have no true parcel in the pool at all.

### 4.5 Throughput

`01e` is rate-limited by Planetary Computer, not CPU. Two measured points,
both at 32 workers in one job:

- 880 candidates (holdout): **85/min**, one 429
- 6,555 candidates (training): **39/min**, 429s throughout

Same worker count, half the rate — an account-level *sustained* limit. So N
concurrent array tasks does **not** give N× throughput, and past some point
more concurrency buys only backoff. National `--all` is 240,546 candidates,
i.e. ~100 hours at 39/min. Find the throttle empirically before committing.

Two unexplained ~17-minute stalls appeared in the training run, not
accounted for by logged retries (`with_retry` caps near 2 min). Probable
cause: a hung HTTP read — `rasterio.open` and the windowed read have no
timeout, so a stalled connection blocks a worker indefinitely. Worth fixing
before long runs.

---

## 5. Current status

**Running:** `01e_run_od_candidates_array.slurm`, 48 states at `%6`,
`WORKERS=8`. Writes to `data/od_features_candidates/candidates/state=XX/`.
No merge needed — tasks never share files. Resume is per (plant, parcel);
restart a timed-out index **without** `NORESUME`.

**Ready to run when it finishes:**

```
sbatch 05b_rerank_candidates.slurm                    # with OD
sbatch --export=NOOD="1" 05b_rerank_candidates.slurm  # deployable now
```

Then point `10_build_review_queue.py` at
`stage2_candidates_reranked.parquet`. Both scores are preserved per
candidate so a reviewer can compare Stage 2a against the re-rank, and plants
resolved by fallback carry `rerank_fallback = True`.

### Not done

1. **Holdout evaluation of the re-ranker.** OD for the holdout's 880
   candidates already exists. `12_score_holdout.py` does not.
2. **`10` is not yet pointed at the reranked table.**
3. **The freshness check's per-partition blind spot** (§3.2).
4. **The 16 quarantined 200m labels** (50 boxes) remain unresolved.
5. **mAP50-95 / box drift** (§2.4) — possibly the same root cause.
6. **`oxidation_pond`** worth revisiting now that the 500m objection is void.

### Ordering advice

`08` says the true parcel is inside the k=18 ring 91% of the time but
reaches the top 20 only 64–81% of the time. Ranking now recovers most of
what reaches the pool. **The larger remaining gain is in candidate
generation, not ranking** — that is where the next round's effort is best
spent.
