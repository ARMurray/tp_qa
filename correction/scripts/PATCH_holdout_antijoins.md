# Patch: holdout anti-joins in `03`, `04`, `06`, `07`

Apply in the **same commit** as `09_build_holdout.py`. A manifest nothing reads
is worse than no manifest — it looks like protection while holdout plants leak
into training anyway.

All four are the same two-line shape: import the helper, call it immediately
after loading the training table and before any feature selection or splitting.
`exclude_holdout()` prints what it removed, so the row counts in your logs are
the evidence it ran.

Every script also gets `--allow-no-holdout` for deliberate pre-holdout runs.
Without it the helper raises when the manifest is missing, which is the intended
behavior: a run that stops beats a run that silently trains on the evaluation
set.

---

## All four scripts — add to the argparse block

```python
    ap.add_argument("--allow-no-holdout", action="store_true",
                     help="proceed even if the holdout manifest is missing. Only "
                          "for deliberate pre-holdout runs -- normally a missing "
                          "manifest should stop the job.")
```

---

## `03_train_stage1.py`

Add near the other imports (after `import config as C`):

```python
from holdout import exclude_holdout
```

Find (line ~55):

```python
    s1 = pd.read_parquet(C.FEATURES_OUTPUT_DIR / "14_stage1_training.parquet")
```

Add immediately after it:

```python
    s1 = exclude_holdout(s1, "stage1", allow_missing=args.allow_no_holdout)
```

Must land **before** the coordinate merge on line ~70, so the spatial folds are
built from training plants only.

---

## `04_train_stage2.py`

Import:

```python
from holdout import exclude_holdout
```

Find (line ~102):

```python
    s2 = pd.read_parquet(C.FEATURES_OUTPUT_DIR / "15_stage2_training.parquet")
```

Add immediately after:

```python
    s2 = exclude_holdout(s2, "stage2a", allow_missing=args.allow_no_holdout)
```

Before the `print(f"  Plants: ...")` on the next line, so the reported count
reflects the post-exclusion set. Also before the balancing step — excluding
after balancing would leave the class ratio computed against plants that then
get dropped.

---

## `06_build_stage2b_training.py`

This one excludes at **write** rather than at load, since `06` assembles from
several sources rather than loading one training table.

Import:

```python
from holdout import exclude_holdout
```

Find (line ~221):

```python
    stage2b.to_parquet(out_path, index=False)
```

Replace with:

```python
    # Exclude holdout plants before writing, so 07 receives a table that is
    # already clean. 07 excludes again -- belt and braces, and the second call
    # is a no-op that costs nothing.
    stage2b = exclude_holdout(stage2b, "stage2b-build",
                              allow_missing=args.allow_no_holdout)
    stage2b.to_parquet(out_path, index=False)
```

---

## `07_train_stage2b.py`

Import:

```python
from holdout import exclude_holdout
```

Find:

```python
    s2b = pd.read_parquet(C.FEATURES_OUTPUT_DIR / "16_stage2b_training.parquet")
    print(f"  Rows: {len(s2b)}")
```

Replace with:

```python
    s2b = pd.read_parquet(C.FEATURES_OUTPUT_DIR / "16_stage2b_training.parquet")
    s2b = exclude_holdout(s2b, "stage2b", allow_missing=args.allow_no_holdout)
    print(f"  Rows: {len(s2b)}")
```

Before the label counts and the paired-plant check, so every printed statistic
describes the actual training set.

---

## Verifying it works

After patching, `09` then a retrain. Each log should show a `[holdout]` line:

```
  [holdout] stage2b: removed 37 rows / 24 plants (448 -> 411)
```

Then confirm the exclusion is real rather than cosmetic:

```python
import pandas as pd
m = set(pd.read_parquet("/work/GRDVULN/correction/data/holdout/holdout_manifest.parquet")["CWNS_ID"].astype(str))
t = pd.read_parquet("/work/GRDVULN/correction/data/features/16_stage2b_training.parquet")
print("leaked:", len(m & set(t["CWNS_ID"].astype(str))))   # must be 0
```

A non-zero result means one of the four calls is in the wrong place.

Expect Stage 2b to lose roughly 8% of its rows to the holdout — painful at ~448,
but the alternative is having no trustworthy way to tell whether round 3 beat
round 2. The review loop is what refills it.
