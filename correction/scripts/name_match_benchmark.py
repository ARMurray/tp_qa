"""
name_match_benchmark.py
=======================
Read-only benchmark: how well does comparing a candidate parcel's OWNER with
the plant's FACILITY NAME pick out the true parcel, method by method?

    sbatch name_match_benchmark.slurm
    -> correction/diagnostics/name_match_benchmark_<host>_<stamp>.txt
       correction/diagnostics/name_match_pairs_<stamp>.csv

WHY (2026-09-25)
    Round 3's reviewer notes kept finding the answer in the owner field:
    'Suffolk county sewer district NO 2Tallmadge woods' for the plant
    'Suffolk (Co) SCSD #2 Tallmadge Woods STP', ranked 4th; an owner
    containing 'Jamestown' for the Jamestown plant, ranked 2nd. The pipeline
    never compares owner with the plant NAME -- 02's add_name_matching only
    checks the owner against keyword lists and the plant's town / place /
    county names. This measures what a name feature would be worth before
    anything is built into 02.

WHAT IT MEASURES
    Pools: 17_rerank_training.parquet -- each training correction's top-20
    Stage 2a candidates, labelled 1 for the true parcel. Holdout plants are
    already excluded from that table, and this script never reads the
    holdout, so choosing a method here cannot leak into 12's score.

    For every method, candidates are ranked by name similarity alone within
    each pool (ties broken by Stage 2a's rank), and reported:
      recall@1 / @3, MRR   over pools that contain the true parcel
      rescue rate          recall@1 among pools where Stage 2a's own #1 was
                           WRONG -- the feature's value is where the current
                           model fails, not where it already succeeds
      damage rate          how often it moves a pool Stage 2a got right to
                           a wrong #1 if used alone
      pair AUC             true-vs-competitor separation across all pairs
    and the COVERAGE that bounds all of it: how many candidates, and how many
    true parcels, have any owner text at all.

    Used alone is the pessimistic case. In the models it would be one feature
    among ~70, so read the rescue rate as its ceiling and the damage rate as
    what the model must learn to discount.

THE METHODS
    fuzzy_raw        rapidfuzz token_set_ratio on lower-cased text. Baseline.
    fuzzy_norm       the same after domain normalisation: abbreviations
                     expanded (STP, WWTP, SD, Co, Twp, MUA, PSD ...),
                     punctuation dropped, letter/digit runs split
                     ('2Tallmadge' -> '2 tallmadge').
    distinct_idf     overlap of the DISTINCTIVE tokens only -- generic words
                     (sewer, district, city, of ...) removed, the rest
                     weighted by rarity across every facility name and owner
                     in the pools, fuzzy-matched token to token. Built for
                     'Tallmadge Woods' and 'Jamestown': the rare word is the
                     signal.
    char_tfidf       cosine similarity of character 3-5-gram TF-IDF vectors
                     over normalised text. Robust to spelling and joined words.
    emb_minilm       sentence-transformers all-MiniLM-L6-v2 cosine
    emb_bge          BAAI/bge-small-en-v1.5 cosine
                     Both run locally on CPU; the model is downloaded once
                     into $ROOT/.hf_cache. Skipped, with a note, if
                     sentence-transformers is not installed (INSTALL=1 in
                     the wrapper installs it).

    No external API is called. An LLM judge for hard pairs is a later step,
    only if the local methods fall short.

READ-ONLY: writes nothing but its report and the pairs CSV.
"""
import argparse
import datetime as dt
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from collect_diagnostics import Report, sec_environment

OUT_DIR = C.ROOT / "diagnostics"
FACILITIES_PATH = C.CWNS_DIR / "FACILITIES.txt"
RERANK_PATH = C.FEATURES_OUTPUT_DIR / "17_rerank_training.parquet"
PARCELS_PATH = C.FEATURES_OUTPUT_DIR / "10_parcel_features.parquet"

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
# Order matters only for multi-word expansions; single tokens map 1:1.
# Deliberately conservative: 'co' -> 'county' is right in facility names
# ('Suffolk (Co)') but can be 'company' in an owner, which is why every
# method is also scored on raw text -- if an expansion hurts, it shows.
ABBREV = {
    "wwtp": "wastewater treatment plant", "wwtf": "wastewater treatment facility",
    "wtp": "treatment plant", "stp": "sewage treatment plant",
    "wpcp": "water pollution control plant", "wpcf": "water pollution control facility",
    "wrf": "water reclamation facility", "wrrf": "water resource recovery facility",
    "wrp": "water reclamation plant", "potw": "publicly owned treatment works",
    "sd": "sanitary district",
    "msd": "metropolitan sewer district", "psd": "public service district",
    "mud": "municipal utility district", "mua": "municipal utilities authority",
    "ua": "utilities authority", "sa": "sewer authority", "wsa": "water sewer authority",
    "co": "county", "cnty": "county", "cty": "city", "twp": "township",
    "vlg": "village", "vill": "village", "boro": "borough", "bor": "borough",
    "auth": "authority", "dist": "district", "dst": "district",
    "muni": "municipal", "mun": "municipal", "reg": "regional", "regl": "regional",
    "trtmt": "treatment", "trmt": "treatment", "trt": "treatment",
    "plt": "plant", "fac": "facility", "wtr": "water", "swr": "sewer",
    "dept": "department", "comm": "commission", "commrs": "commissioners",
    "st": "saint", "mt": "mount", "ft": "fort", "no": "number",
}

# Words that say "this is a utility / a government" but not WHICH one. Kept
# for the fuzzy methods (they do separate a sewer district from a farm) but
# dropped for distinct_idf, whose whole job is the identifying word.
GENERIC = set("""
    the of and at for in on to a an
    wastewater waste water sewer sewage sewerage sanitary sanitation treatment
    plant plants facility facilities works reclamation recovery resource
    pollution control lagoon lagoons pond ponds system systems utility utilities
    authority district department commission commissioners board service services
    public publicly owned municipal metropolitan regional joint
    city town township village borough county county's state parish
    inc llc corp corporation company co ltd lp trust
    number east west north south upper lower new old
""".split())


def normalise(s: str) -> str:
    s = str(s).lower()
    s = s.replace("&", " and ").replace("#", " number ")
    s = re.sub(r"([a-z])(\d)", r"\1 \2", s)       # 'no2' -> 'no 2'
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s)       # '2tallmadge' -> '2 tallmadge'
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = []
    for t in s.split():
        toks.extend(ABBREV.get(t, t).split())
    return " ".join(toks)


def distinctive(s_norm: str) -> list[str]:
    return [t for t in s_norm.split() if t not in GENERIC and not t.isdigit() and len(t) > 1]


# ---------------------------------------------------------------------------
# Methods -- each returns one score per row of `pairs`, higher = more alike
# ---------------------------------------------------------------------------
def m_fuzzy_raw(pairs, fuzz):
    return np.array([fuzz.token_set_ratio(a.lower(), b.lower()) / 100.0
                     for a, b in zip(pairs["facility_name"], pairs["owner"])])


def m_fuzzy_norm(pairs, fuzz):
    return np.array([fuzz.token_set_ratio(a, b) / 100.0
                     for a, b in zip(pairs["fac_norm"], pairs["own_norm"])])


def m_distinct_idf(pairs, fuzz):
    """Rarity-weighted share of the facility name's distinctive tokens found
    (fuzzily, ratio >= 88) in the owner's distinctive tokens."""
    docs = pd.concat([pairs["fac_norm"].drop_duplicates(),
                      pairs["own_norm"].drop_duplicates()])
    df_count: dict[str, int] = {}
    for d in docs:
        for t in set(distinctive(d)):
            df_count[t] = df_count.get(t, 0) + 1
    n_docs = max(len(docs), 1)
    idf = {t: np.log((1 + n_docs) / (1 + c)) + 1.0 for t, c in df_count.items()}

    out = np.zeros(len(pairs))
    for i, (a, b) in enumerate(zip(pairs["fac_norm"], pairs["own_norm"])):
        fa, ob = distinctive(a), distinctive(b)
        if not fa or not ob:
            continue
        total = sum(idf.get(t, 1.0) for t in fa)
        hit = 0.0
        for t in fa:
            best = max(fuzz.ratio(t, u) for u in ob)
            if best >= 88:
                hit += idf.get(t, 1.0) * best / 100.0
        out[i] = hit / total if total else 0.0
    return out


def m_char_tfidf(pairs, _fuzz):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    vec.fit(pd.concat([pairs["fac_norm"], pairs["own_norm"]]).drop_duplicates())
    A = vec.transform(pairs["fac_norm"])
    B = vec.transform(pairs["own_norm"])
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()   # rows are L2-normalised


def m_embedding(model_name):
    def run(pairs, _fuzz):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device="cpu")
        texts = pd.concat([pairs["facility_name"], pairs["owner"]]).drop_duplicates().tolist()
        emb = model.encode(texts, batch_size=256, normalize_embeddings=True,
                           show_progress_bar=False)
        lookup = dict(zip(texts, emb))
        A = np.stack([lookup[t] for t in pairs["facility_name"]])
        B = np.stack([lookup[t] for t in pairs["owner"]])
        return (A * B).sum(axis=1)
    return run


METHODS = {
    "fuzzy_raw": m_fuzzy_raw,
    "fuzzy_norm": m_fuzzy_norm,
    "distinct_idf": m_distinct_idf,
    "char_tfidf": m_char_tfidf,
    "emb_minilm": m_embedding("sentence-transformers/all-MiniLM-L6-v2"),
    "emb_bge": m_embedding("BAAI/bge-small-en-v1.5"),
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def rank_within_pools(df: pd.DataFrame, score_col: str) -> pd.Series:
    """1 = best. A candidate with no owner text scores 0 -- the same as a
    completely dissimilar owner, never better -- and ties are broken by Stage
    2a's rank: 'use the name when it says something, otherwise keep what the
    model had'. Negative cosines clip to 0 for the same reason."""
    s = df[score_col].fillna(0.0).clip(lower=0.0)
    order = df.assign(_s=s).sort_values(["CWNS_ID", "_s", "stage2a_rank"],
                                        ascending=[True, False, True])
    return order.groupby("CWNS_ID").cumcount().add(1).reindex(df.index)


def summarise(df: pd.DataFrame, score_col: str) -> dict:
    from sklearn.metrics import roc_auc_score

    r = rank_within_pools(df, score_col)
    true = df[df["label"] == 1].assign(r=r[df["label"] == 1])
    s2a_right = set(df.loc[(df["label"] == 1) & (df["stage2a_rank"] == 1), "CWNS_ID"])
    wrong_pools = true[~true["CWNS_ID"].isin(s2a_right)]
    right_pools = true[true["CWNS_ID"].isin(s2a_right)]
    have = df[score_col].notna()
    auc = (roc_auc_score(df.loc[have, "label"], df.loc[have, score_col])
           if df.loc[have, "label"].nunique() == 2 else float("nan"))
    return dict(
        pools=len(true),
        r1=(true["r"] == 1).mean(), r3=(true["r"] <= 3).mean(),
        mrr=(1.0 / true["r"]).mean(),
        rescue=(wrong_pools["r"] == 1).mean() if len(wrong_pools) else float("nan"),
        n_wrong=len(wrong_pools),
        damage=(right_pools["r"] != 1).mean() if len(right_pools) else float("nan"),
        n_right=len(right_pools),
        auc=auc,
    )


# ---------------------------------------------------------------------------
def load_pairs(r) -> pd.DataFrame:
    import duckdb  # noqa: F401  (C.duckdb_connect imports it)

    if not FACILITIES_PATH.exists():
        raise SystemExit(f"{FACILITIES_PATH} not found. Commit FACILITIES.txt from the "
                         f"Sewersheds repo to correction/data/cwns/ and pull.")
    try:
        fac = pd.read_csv(FACILITIES_PATH, dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        fac = pd.read_csv(FACILITIES_PATH, dtype=str, encoding="latin1")
    fac = (fac[["CWNS_ID", "FACILITY_NAME"]].dropna().drop_duplicates(subset="CWNS_ID")
           .rename(columns={"FACILITY_NAME": "facility_name"}))
    r.p(f"  facility names: {len(fac):,} plants")

    pools = pd.read_parquet(RERANK_PATH, columns=["CWNS_ID", "ll_uuid", "label",
                                                  "stage2a_rank"])
    pools["CWNS_ID"] = pools["CWNS_ID"].astype(str)
    pools["ll_uuid"] = pools["ll_uuid"].astype(str)
    has_pos = pools.groupby("CWNS_ID")["label"].transform("max") == 1
    r.p(f"  re-rank pools: {pools['CWNS_ID'].nunique()} plants, "
        f"{int(has_pos.groupby(pools['CWNS_ID']).first().sum())} with the true "
        f"parcel in the top 20 (only those can be scored)")
    pools = pools[has_pos]

    con = C.duckdb_connect(spatial=False)
    con.register("want", pools[["ll_uuid"]].drop_duplicates())
    owners = con.execute(f"""
        SELECT p.ll_uuid, p.owner
        FROM read_parquet('{PARCELS_PATH.as_posix()}') p
        JOIN want w ON w.ll_uuid = p.ll_uuid
    """).df().drop_duplicates(subset="ll_uuid")
    con.close()

    df = pools.merge(owners, on="ll_uuid", how="left") \
              .merge(fac, on="CWNS_ID", how="left")
    df["owner"] = df["owner"].fillna("").astype(str).str.strip()
    df["facility_name"] = df["facility_name"].fillna("").astype(str).str.strip()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS),
                    help="comma-separated subset of: " + ", ".join(METHODS))
    ap.add_argument("--examples", type=int, default=15,
                    help="how many rescued / missed examples to print per best method")
    args = ap.parse_args()
    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]

    os.environ.setdefault("HF_HOME", str(C.ROOT / ".hf_cache"))

    r = Report()
    r.p("tp_qa owner-name vs facility-name benchmark")
    r.p("Generated by name_match_benchmark.py -- read-only.")
    r.section("1. ENVIRONMENT", sec_environment)

    r.h("2. DATA")
    df = load_pairs(r)
    have_name = df["facility_name"].str.len() > 0
    have_owner = df["owner"].str.len() > 0
    true = df["label"] == 1
    r.p(f"  candidate rows scored        : {len(df):,} across {df['CWNS_ID'].nunique()} plants")
    r.p(f"  plants with a facility name  : {df.loc[have_name, 'CWNS_ID'].nunique()}")
    r.p(f"  candidates with owner text   : {int(have_owner.sum()):,} ({have_owner.mean():.1%})")
    r.p(f"  TRUE parcels with owner text : {int((true & have_owner).sum())}/{int(true.sum())} "
        f"({(true & have_owner).sum() / max(true.sum(), 1):.1%})  <- ceiling for every method")

    scorable = have_name & have_owner
    df["fac_norm"] = df["facility_name"].map(normalise)
    df["own_norm"] = df["owner"].map(normalise)
    r.p("\n  normalisation examples:")
    for _, x in df[true & scorable].head(6).iterrows():
        r.p(f"    {x['facility_name'][:48]:<48} -> {x['fac_norm'][:60]}")
        r.p(f"    {x['owner'][:48]:<48} -> {x['own_norm'][:60]}")

    try:
        from rapidfuzz import fuzz
    except ImportError:
        raise SystemExit("rapidfuzz is not installed -- resubmit with "
                         "--export=INSTALL=1 (see the .slurm header).")

    r.h("3. RESULTS -- each method used ALONE to rank the pool")
    r.p("  recall@1/@3 and MRR over pools containing the true parcel. rescue = "
        "recall@1 where Stage 2a's own #1 was WRONG;")
    r.p("  damage = share of pools Stage 2a had right that this method would "
        "get wrong on its own.\n")
    # Reference: Stage 2a's own order, expressed as a score (higher = better).
    base = summarise(df.assign(_b=1.0 / df["stage2a_rank"].astype(float)), "_b")
    rows = [dict(method="stage2a (reference)", **base)]
    sub = df[scorable]
    for name in wanted:
        if name not in METHODS:
            r.p(f"  unknown method {name!r} -- skipped")
            continue
        try:
            vals = METHODS[name](sub, fuzz)
        except ImportError as e:
            r.p(f"  {name}: skipped -- {e}. INSTALL=1 installs sentence-transformers.")
            continue
        except Exception as e:
            r.p(f"  {name}: FAILED -- {type(e).__name__}: {e}")
            continue
        df[name] = np.nan
        df.loc[sub.index, name] = vals
        rows.append(dict(method=name, **summarise(df, name)))

    res = pd.DataFrame(rows)
    r.p(f"  {'method':<22}{'r@1':>7}{'r@3':>7}{'MRR':>7}{'rescue':>9}{'damage':>9}{'AUC':>7}")
    for x in res.itertuples():
        r.p(f"  {x.method:<22}{x.r1:>7.1%}{x.r3:>7.1%}{x.mrr:>7.3f}"
            f"{x.rescue:>9.1%}{x.damage:>9.1%}{x.auc:>7.3f}")
    r.p(f"\n  pools: {int(res['pools'].iloc[0])}  |  Stage 2a wrong in "
        f"{int(res['n_wrong'].iloc[0])}, right in {int(res['n_right'].iloc[0])}")
    r.p("  Note: these are TRAINING plants, where Stage 2a's reference row is "
        "optimistic (it trained on them).")

    scored = res[res["method"] != "stage2a (reference)"]
    if len(scored):
        best = scored.sort_values(["rescue", "auc"], ascending=False)["method"].iloc[0]
        r.h(f"4. EXAMPLES -- best rescuer: {best}")
        rk = rank_within_pools(df, best)
        t = df[true].assign(r=rk[true])
        wrong = t[t["stage2a_rank"] != 1]
        for title, rows_ in (("RESCUED (Stage 2a wrong, name gets #1)", wrong[wrong["r"] == 1]),
                             ("STILL MISSED (Stage 2a wrong, name does not get #1)", wrong[wrong["r"] != 1])):
            r.p(f"\n  {title}: {len(rows_)}")
            for _, x in rows_.head(args.examples).iterrows():
                r.p(f"    {x['CWNS_ID']}  s2a#{int(x['stage2a_rank']):<2} name#{int(x['r']):<2}"
                    f"  {best}={x[best]:.2f}")
                r.p(f"      plant: {x['facility_name'][:90]}")
                r.p(f"      owner: {x['owner'][:90] or '(no owner text)'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    host = platform.node().split(".")[0]
    out = OUT_DIR / f"name_match_benchmark_{host}_{stamp}.txt"
    out.write_text(r.buf.getvalue(), encoding="utf-8")
    keep = ["CWNS_ID", "ll_uuid", "label", "stage2a_rank", "facility_name", "owner"] + \
        [m for m in wanted if m in df.columns]
    csv = OUT_DIR / f"name_match_pairs_{stamp}.csv"
    df[keep].to_csv(csv, index=False)

    print(r.buf.getvalue())
    print(f"\nWRITTEN: {out}\n         {csv}")
    print("\nCommit and push, from a login node:")
    print("  cd /work/GRDVULN/tp_qa")
    print("  git add correction/diagnostics/")
    print(f'  git commit -m "name match benchmark {stamp}"')
    print("  git push")


if __name__ == "__main__":
    main()
