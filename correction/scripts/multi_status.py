"""
multi_status.py -- one text file that says where the multi-source run stands
=============================================================================
Writes correction/diagnostics/multi_status.txt, which is tracked in git (diagnostics/ is not ignored), so the
status of every step travels back with `git pull` and can be read without HPC access:

    bash correction/scripts/push_diagnostics.sh          # refresh it, commit diagnostics/, push

Every job also refreshes it when it ends (_latest_log.sh calls this, at most one copy at a time), so after any step
the file is current; only the commit and push are manual.

What it reports, in run order: environment; 01a window sizes against the single-seed run; 01f seeds, sites and
how seeds matched parcels; 01e seed-site detection coverage; 02's Stage 1 and Stage 2 tables; the models and their
thresholds; 05's decisions; 13's provenance; and the last lines of any job log whose exit status was not 0.

Reads parquet footers where it can and small columns otherwise, so it takes seconds. Every section is wrapped: a
missing or odd file produces a line saying so, never a crash. Does nothing when TPQA_SEED_MODE=cwns_only.

Usage:  python multi_status.py [--quiet]
"""
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

OUT = C.ROOT / "diagnostics" / "multi_status.txt"
LOG_DIR = C.ROOT / "diagnostics" / "latest_logs"


def rows_of(path: Path) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def read_cols(path: Path, cols):
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(path).schema_arrow.names)
    return pd.read_parquet(path, columns=[c for c in cols if c in have])


def vc(s: pd.Series, n=8) -> str:
    return ", ".join(f"{k}: {v:,}" for k, v in s.value_counts(dropna=False).head(n).items()) or "(none)"


def age(path: Path) -> str:
    h = (time.time() - path.stat().st_mtime) / 3600
    return f"{h / 24:.1f} d ago" if h >= 48 else f"{h:.1f} h ago"


# --------------------------------------------------------------------------- sections
def sec_env(L):
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(C.ROOT), *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:
            return f"({e})"
    L.append(f"generated   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"seed mode   : {C.SEED_MODE}   suffix '{C.SFX}'")
    L.append(f"git commit  : {git('rev-parse', '--short', 'HEAD')}  ({git('log', '-1', '--format=%cd %s')})")
    L.append(f"seeds file  : {C.SEEDS_CSV}  " + (f"{C.SEEDS_CSV.stat().st_size / 1e6:.1f} MB" if C.SEEDS_CSV.exists()
                                               else "MISSING -- git pull"))
    for name, p in [("training gpkg", C.TRAINING_GPKG), ("holdout manifest", C.DATA_DIR / "holdout" / "holdout_manifest.parquet"),
                    ("detector best.pt", next(iter(sorted(C.OD_MODEL_DIR.rglob("*.pt"))), C.OD_MODEL_DIR / "best.pt"))]:
        L.append(f"{name:<12}: " + (f"{age(p)}" if p.exists() else "MISSING") + f"   {p}")


def sec_01a(L):
    d, base = C.NLCD_OUTPUT_DIR, C.DATA_DIR / "nlcd_features"
    files = sorted(d.glob(f"nlcd_*_k{C.K_RINGS}.parquet")) if d.exists() else []
    L.append(f"{d}: {len(files)} state file(s)")
    if not files:
        return
    rows = []
    for f in files:
        st = f.name.split("_")[1]
        b = base / f.name
        rows.append((st, rows_of(f), rows_of(b) if (b.exists() and C.MULTI) else np.nan))
    t = pd.DataFrame(rows, columns=["state", "parcels", "single_seed"])
    t["ratio"] = t["parcels"] / t["single_seed"]
    L.append(f"total parcels {int(t['parcels'].sum()):,}; newest file {age(max(files, key=lambda p: p.stat().st_mtime))}")
    if t["single_seed"].notna().any():
        r = t["ratio"].dropna()
        L.append(f"vs single-seed run ({int(t['single_seed'].sum()):,}): ratio median {r.median():.2f}, "
                 f"min {r.min():.2f}, max {r.max():.2f}  (the recall check expected ~1.0-1.3)")
        L.append("largest growth: " + ", ".join(f"{s} {x:.2f}" for s, x in t.nlargest(5, 'ratio')[["state", "ratio"]].itertuples(index=False)))
    L.append("states: " + " ".join(t["state"]))


def sec_01f(L):
    d = C.SEED_SITES_DIR
    dirs = sorted(d.glob("state=*")) if d.exists() else []
    L.append(f"{d}: {len(dirs)} state dir(s)")
    if not dirs:
        return
    sp = pd.concat([read_cols(x / "seed_parcels.parquet", ["CWNS_ID", "match_kind", "snap_m", "site_id", "source"]) for x in dirs])
    si = pd.concat([read_cols(x / "seed_sites.parquet", ["CWNS_ID", "site_id", "n_origins", "n_origins_all", "n_seeds",
                                                         "site_n_parcels", "oversized", "snapped", "has_cwns_point", "n_sites",
                                                         "d_nearest_other_site_m", "d_to_cwns_point_m"])
                    for x in dirs])
    L.append(f"seeds {len(sp):,} for {sp['CWNS_ID'].nunique():,} plants; sites {len(si):,} for {si['CWNS_ID'].nunique():,} plants")
    L.append(f"seed -> parcel match: {vc(sp['match_kind'])}")
    L.append("  by source (share with no parcel): " + ", ".join(
        f"{s} {(g['match_kind'] == 'none').mean():.1%}" for s, g in sp.groupby("source")))
    sn = sp.loc[sp["match_kind"] == "snapped", "snap_m"]
    if len(sn):
        L.append(f"  snapped seeds: median {sn.median():.0f} m, max {sn.max():.0f} m")
    L.append(f"plants with no site: {sp['CWNS_ID'].nunique() - si['CWNS_ID'].nunique():,}")
    L.append(f"sites per plant: {si.groupby('CWNS_ID').size().value_counts().sort_index().head(8).to_dict()}")
    L.append(f"independent origins per site: {si['n_origins'].value_counts().sort_index().to_dict()}  "
             f"(counting copies: {si['n_origins_all'].value_counts().sort_index().to_dict()})")
    if "d_nearest_other_site_m" in si:
        multi = si[si["n_sites"] > 1]
        d = multi["d_nearest_other_site_m"].dropna()
        edges = [(0, 50), (50, 100), (100, 250), (250, 500), (500, 1000), (1000, 1e9)]
        L.append(f"distance to the plant's nearest OTHER site ({len(d):,} sites in multi-site plants) -- close ones may be one "
                 f"facility split across owners:")
        L.append("  " + "   ".join(f"{a:g}-{b:g} m: {((d >= a) & (d < b)).mean():.0%}" for a, b in edges).replace("-1e+09", "+"))
        nc = si.loc[~si["has_cwns_point"] & si["d_to_cwns_point_m"].notna(), "d_to_cwns_point_m"]
        if len(nc):
            L.append(f"sites WITHOUT the CWNS point ({len(nc):,}): distance to the CWNS point median {nc.median():,.0f} m; "
                     + ", ".join(f"<{t} m {(nc < t).mean():.0%}" for t in (250, 500, 1000, 5000)))
    L.append(f"sites holding the CWNS point: {int(si['has_cwns_point'].sum()):,}; oversized parcel: {int(si['oversized'].sum()):,}; "
             f"merged parcels (site_n_parcels>1): {int((si['site_n_parcels'] > 1).sum()):,}")


def sec_01e(L):
    d = C.OD_SEED_SITES_DIR / "candidates"
    files = sorted(d.glob("state=*/part-*.parquet")) if d.exists() else []
    L.append(f"{d}: {len(files)} part file(s)")
    if not files:
        return
    files = sorted(files, key=lambda f: f.stat().st_mtime)  # newest row wins, as in 02's reader
    od = pd.concat([read_cols(f, ["CWNS_ID", "ll_uuid", "od_ran", "od_has_detection", "parcel_capped",
                                  "od_fetch_error"])
                    .assign(state=f.parent.name[6:]) for f in files]).drop_duplicates(["CWNS_ID", "ll_uuid"], keep="last")
    ran = od["od_ran"].fillna(False).astype(bool)
    L.append(f"parcels examined {len(od):,}; od_ran {ran.mean():.1%}; "
             f"something detected {od['od_has_detection'].fillna(False).astype(bool).mean():.1%}")
    if (~ran).any():
        per_fail = od[~ran].groupby("state").size().sort_values(ascending=False)
        L.append(f"od_ran False (every tile failed; a rerun retries them): {int((~ran).sum()):,}  -- top: "
                 + ", ".join(f"{s} {n}" for s, n in per_fail.head(8).items()))
        if "od_fetch_error" in od:
            # error kinds, numbers and URLs blanked so one cause is one line
            e = od.loc[~ran, "od_fetch_error"].dropna().astype(str) \
                .str.replace(r"https?://\S+", "<url>", regex=True).str.replace(r"\d+(\.\d+)?", "#", regex=True) \
                .str.slice(0, 110)
            L.append(f"  first tile error, {len(e):,} of them recorded (rows written before 2026-11-04 have none):")
            L += [f"    {n:>6,}  {k}" for k, n in e.value_counts().head(6).items()]
    if "parcel_capped" in od:
        L.append(f"oversized parcels capped to one tile around the seed: "
                 f"{int(od['parcel_capped'].fillna(False).astype(bool).sum()):,}")
    sd = C.SEED_SITES_DIR
    if sd.exists():
        need = pd.concat([read_cols(x / "seed_sites.parquet", ["CWNS_ID", "rep_ll_uuid"]).assign(state=x.name[6:])
                          for x in sorted(sd.glob("state=*"))])
        have = set(zip(od["CWNS_ID"].astype(str), od["ll_uuid"].astype(str)))
        need["done"] = [(c, u) in have for c, u in zip(need["CWNS_ID"], need["rep_ll_uuid"])]
        L.append(f"coverage of seed sites: {int(need['done'].sum()):,} / {len(need):,} = {need['done'].mean():.1%}")
        per = need.groupby("state")["done"].mean()
        L.append("states under 95%: " + (", ".join(f"{s} {v:.0%}" for s, v in per[per < 0.95].items()) or "none"))
    pts = sorted(C.OD_MODEL_DIR.rglob("*.pt"), key=lambda p: p.stat().st_mtime) if C.OD_MODEL_DIR.exists() else []
    if pts:
        stale = sum(f.stat().st_mtime < pts[-1].stat().st_mtime for f in files)
        L.append(f"part files older than best.pt (ignored downstream): {stale}")


def sec_02(L):
    d = C.FEATURES_OUTPUT_DIR
    L.append(f"{d}")
    for n in ("05_plant_features", "10_parcel_features", "14_stage1_training", "15_stage2_training"):
        p = d / f"{n}.parquet"
        L.append(f"  {n:<22}" + (f"{rows_of(p):>11,} rows   {age(p)}" if p.exists() else "missing"))
    p = d / "14_stage1_training.parquet"
    if p.exists():
        s = read_cols(p, ["CWNS_ID", "site_id", "class", "has_cwns_point", "n_origins", "od_ran", "oversized", "name_match_score"])
        L.append(f"Stage 1: {len(s):,} site rows, {s['CWNS_ID'].nunique():,} plants; classes {vc(s['class'])}")
        corr = s.groupby("CWNS_ID")["class"].apply(lambda c: int((c == "Correct").sum()))
        L.append(f"  Correct sites per plant: {corr.value_counts().sort_index().to_dict()}")
        if "has_cwns_point" in s:
            hc = s["has_cwns_point"].fillna(False).astype(bool)
            L.append("  Correct rate when site holds the CWNS point / not: " + " / ".join(
                f"{(g['class'] == 'Correct').mean():.1%} (n={len(g):,})" for g in (s[hc], s[~hc])))
        if "n_origins" in s:
            L.append("  Correct rate by independent origins: " + ", ".join(
                f"{k}: {(g['class'] == 'Correct').mean():.1%}" for k, g in s.groupby("n_origins")))
        if "od_ran" in s:
            L.append(f"  od_ran True on {s['od_ran'].fillna(False).astype(bool).mean():.1%} of site rows")
    p = d / "15_stage2_training.parquet"
    if p.exists():
        s2 = read_cols(p, ["CWNS_ID", "label", "on_seed_parcel", "on_parcel_cwns_point", "on_parcel_icis"])
        pos = s2[s2["label"] == 1]
        L.append(f"Stage 2: {len(s2):,} candidate rows, {s2['CWNS_ID'].nunique():,} plants, {len(pos):,} positives")
        for c in ("on_seed_parcel", "on_parcel_cwns_point", "on_parcel_icis"):
            if c in s2:
                L.append(f"  truth parcel {c}: {pos[c].mean():.1%} of positives vs {s2[c].mean():.2%} of all candidates")


def sec_models(L):
    import joblib
    d = C.MODELS_DIR
    L.append(f"{d}")
    for f in sorted(d.glob("*.joblib")) if d.exists() else []:
        extra = ""
        if "threshold" in f.name:
            try:
                extra = f"= {float(joblib.load(f)):.3f}"
            except Exception:
                pass
        L.append(f"  {f.name:<40}{age(f):>12}  {extra}")
    for stage in ("stage1", "stage2"):
        p = d / f"{stage}_rf_importance.parquet"
        if p.exists():
            imp = pd.read_parquet(p).head(12)
            L.append(f"{stage} top features: " + ", ".join(f"{r.Variable} {r.Importance:.3f}" for r in imp.itertuples()))
    p = d / "stage1_plant_threshold_analysis.parquet"
    if p.exists():
        t = pd.read_parquet(p)
        L.append("Stage 1 plant-level accept table (held-out plants):")
        for q in (0.5, 0.7, 0.85, 0.95):
            r = t.iloc[(t["threshold"] - q).abs().argmin()]
            L.append(f"  score>={r['threshold']:.2f}: {int(r['n_accepted'])} plants accepted, precision {r['precision']:.3f}, "
                     f"capture {r['capture']:.1%}")
    p = d / "stage1_cv_comparison.parquet"
    if p.exists():
        L.append("Stage 1 CV: " + "; ".join(f"{r.cv} {r.roc_auc:.3f}" for r in pd.read_parquet(p).itertuples()))


def sec_05(L):
    d = C.INFERENCE_DIR
    p = d / "plant_summary.parquet"
    L.append(f"{d}")
    if not p.exists():
        L.append("  plant_summary.parquet missing")
        return
    s = read_cols(p, ["CWNS_ID", "trigger_reason", "stage1_route", "stage1_accepted", "multi_pass", "n_sites", "accepted_sources",
                      "n_candidates", "stage1_prob_correct", "STATE_CODE"])
    L.append(f"plant_summary: {len(s):,} plants, {age(p)}")
    L.append(f"  trigger_reason: {vc(s['trigger_reason'])}")
    L.append(f"  route: {vc(s['stage1_route'])}")
    if "multi_pass" in s:
        L.append(f"  multi_pass: {int(s['multi_pass'].fillna(False).sum()):,}   sites per plant: {s['n_sites'].value_counts().sort_index().head(6).to_dict()}")
        acc = s[s["trigger_reason"] == "none"]
        L.append(f"  accepted plants whose site lacks the CWNS point: "
                 f"{int((~acc['accepted_sources'].fillna('').str.contains('cwns_point')).sum()):,} of {len(acc):,}")
        L.append(f"  accepted by sources: {vc(acc['accepted_sources'], 10)}")
    fl = s[s["trigger_reason"] != "none"]
    L.append(f"  flagged plants {len(fl):,}; with candidates {int((fl['n_candidates'] > 0).sum()):,}")
    c = d / "stage2_candidates.parquet"
    if c.exists():
        L.append(f"stage2_candidates: {rows_of(c):,} rows, {age(c)}")
    for n in ("stage1_sites", "plant_summary_reranked", "stage2_candidates_reranked"):
        q = d / f"{n}.parquet"
        L.append(f"  {n:<28}" + (f"{rows_of(q):>10,} rows  {age(q)}" if q.exists() else "missing"))


def sec_13(L):
    p = C.OUTPUT_DIR / "cwns_corrected_locations.parquet"
    L.append(f"{p}")
    if not p.exists():
        L.append("  missing")
        return
    o = read_cols(p, ["CWNS_ID", "status", "loc_basis", "led_by", "loc_sources", "loc_n_origins", "moved_m", "multi_pass", "built"])
    L.append(f"  {len(o):,} plants, built {o['built'].iloc[0] if len(o) else '?'}")
    L.append(f"  status: {vc(o['status'], 12)}")
    if "loc_basis" in o:
        L.append(f"  loc_basis: {vc(o['loc_basis'], 12)}")
        L.append(f"  moved plants led by: {vc(o.loc[o['status'] == 'moved', 'led_by'], 10)}")
        L.append(f"  output point supported by independent origins: {o['loc_n_origins'].value_counts().sort_index().to_dict()}")


def sec_logs(L):
    if not LOG_DIR.exists():
        L.append("(no latest_logs)")
        return
    bad = []
    recent = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    L.append(f"{len(recent)} log(s); newest first (name, age, exit status):")
    for f in recent[:25]:
        try:
            with open(f, errors="replace") as fh:
                head = [next(fh, "") for _ in range(2)]
            st = head[1].split("exit status")[-1].split(",")[0].strip() if "exit status" in head[1] else "?"
        except Exception:
            st = "?"
        L.append(f"  {f.name:<34}{age(f):>12}   exit {st}")
        if st not in ("0", "?"):
            bad.append(f)
    for f in bad[:4]:
        L.append(f"\n--- last 60 lines of {f.name} (exit status not 0) ---")
        L.extend(f.read_text(errors="replace").splitlines()[-60:])


SECTIONS = [("ENVIRONMENT", sec_env), ("01a  candidate windows (seeds as centers)", sec_01a),
            ("01f  seed sites", sec_01f), ("01e --seed-sites  detection", sec_01e),
            ("02   feature tables", sec_02), ("03/04 models", sec_models),
            ("05   inference decisions", sec_05), ("13   corrected output + provenance", sec_13),
            ("JOB LOGS", sec_logs)]


def main():
    if not C.MULTI:
        return
    L = ["multi_status.txt -- written by multi_status.py; see its docstring", ""]
    for title, fn in SECTIONS:
        L += ["=" * 78, title, "=" * 78]
        try:
            fn(L)
        except Exception as e:
            L.append(f"(section failed: {type(e).__name__}: {e})")
            L.extend("    " + ln for ln in traceback.format_exc().splitlines()[-4:])
        L.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text("\n".join(L) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    if "--quiet" not in sys.argv:
        print(f"wrote {OUT}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
