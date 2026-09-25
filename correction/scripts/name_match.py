"""
name_match.py
=============
Owner-name vs facility-name similarity, as model features. Shared by 02
(training tables), 05 (inference), 06b (re-rank training) and 05b (re-rank
inference) so every stage computes exactly the same thing -- a feature that
means one thing in training and another at inference is silent and fatal.

WHY (2026-09-25)
    Round 3's reviewer kept finding the answer in the parcel owner field, and
    the pipeline never compared the owner with the plant's NAME. 02's
    add_name_matching only checks the owner against keyword lists and the
    plant's town / place / county. name_match_benchmark.py measured six
    methods on the 311 training plants whose true parcel is in the top 20:
    the rarity-weighted DISTINCTIVE-TOKEN match below was the only one that
    beat Stage 2a used alone (60.8% vs 58.5% recall@1) while breaking just 6%
    of the pools Stage 2a had right; fuzzy and embedding methods broke ~30%.

THE SCORE (name_match_score, 0..1)
    1. Normalise both strings: lower-case, split letter/digit runs
       ('2Tallmadge' -> '2 tallmadge'), expand domain abbreviations (STP,
       WWTP, SD, Co, Twp, MUA, PSD ...).
    2. Keep only the facility name's DISTINCTIVE tokens -- drop generic words
       like sewer, district, city, of, county. Near every candidate owner is
       a city or a district; 'sewage treatment plant' says nothing about
       WHICH one. 'Tallmadge', 'Jamestown', 'Palestine' do.
    3. Weight each distinctive token by its rarity across all ~31k CWNS
       facility names (IDF). Fixed by FACILITIES.txt alone, so training and
       inference use the same weights no matter which parcels are in view.
    4. A token counts as found if some owner token matches it at rapidfuzz
       ratio >= 88 (spelling, truncation: 'GREATER BADIN WATER & SEWER DI'),
       or if it is an ACRONYM of the owner (see below).
    Score = sum of found tokens' IDF x match strength / sum of all their IDF.

    ACRONYMS: a 3-7 letter token on one side that equals, or is a 3+ letter
    prefix / suffix of, the initials of the other side's words: 'FRWRD' <->
    'Fox River Water Reclamation District', 'BCW&SA' <-> 'Berkeley County
    Water & Sewer Authority'. Measured on 1.3% of training plants -- rare,
    cheap, and it cannot fire where it does not apply.

    name_match_available is False when the plant has no facility name or the
    parcel has no owner text. The score is 0 there, and the flag lets a model
    tell 'no match' from 'nothing to compare'.

THE LIMIT IT CANNOT CROSS
    The name identifies the OWNER, not the parcel. 'FREEPORT WWTP' matches
    'CITY OF FREEPORT' at full strength -- and so does every other parcel the
    city owns in the pool. add_pool_features() therefore adds pool context:
    rank within the pool, gap to the pool's best, and how many candidates
    share the best score. '1 of 1 at max' is a very different signal from
    '1 of 6'.

PERFORMANCE
    Stage 2a scores ~10k candidates per plant. Scoring is per plant against
    an inverted index of the call's unique owner tokens, with rapidfuzz doing
    the fuzzy lookups in C, so cost scales with (plants x distinctive tokens),
    not with candidates.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

FACILITIES_PATH = C.CWNS_DIR / "FACILITIES.txt"
FUZZY_CUTOFF = 88
STOP = {"of", "the", "and", "a", "an", "at", "for", "in", "on", "to"}

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

GENERIC = set("""
    the of and at for in on to a an
    wastewater waste water sewer sewage sewerage sanitary sanitation treatment
    plant plants facility facilities works reclamation recovery resource
    pollution control lagoon lagoons pond ponds system systems utility utilities
    authority district department commission commissioners board service services
    public publicly owned municipal metropolitan regional joint
    city town township village borough county state parish
    inc llc corp corporation company ltd lp trust
    number east west north south upper lower new old main
    collection sewers aka
""".split())


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
def normalise(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).lower().replace("&", " and ").replace("#", " number ")
    s = re.sub(r"([a-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    out = []
    for t in s.split():
        out.extend(ABBREV.get(t, t).split())
    return " ".join(out)


def distinctive(norm: str) -> list[str]:
    seen, out = set(), []
    for t in norm.split():
        if t in GENERIC or t.isdigit() or len(t) < 2 or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def initials(norm: str) -> str:
    return "".join(t[0] for t in norm.split() if t not in STOP and not t.isdigit())


def raw_tokens(s) -> list[str]:
    """Pre-expansion tokens, for acronym matching: 'BCW&SA' must stay 'bcwsa'
    here rather than become 'bcw and sewer authority'."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    s = str(s).lower().replace("&", "")
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def acronym_hit(token: str, inits: str) -> bool:
    """token (3-7 letters) is the other side's initials, or they share a 3+
    letter prefix/suffix run."""
    if not (3 <= len(token) <= 7) or not token.isalpha() or len(inits) < 3:
        return False
    if token == inits:
        return True
    short, long_ = sorted((token, inits), key=len)
    return len(short) >= 3 and (long_.startswith(short) or long_.endswith(short))


# ---------------------------------------------------------------------------
# Facility names and IDF
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def load_facility_names(path: Path = FACILITIES_PATH) -> dict[str, str]:
    """CWNS_ID -> FACILITY_NAME. Raises if the file is missing: a model
    trained with these features and scored without them would see every
    plant as 'no name available', silently."""
    key = ("names", str(path))
    if key in _CACHE:
        return _CACHE[key]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. It is tracked in git (correction/data/cwns/"
            f"FACILITIES.txt) -- git pull. Name-match features cannot be built "
            f"without it, and building them from nothing would train a model "
            f"on an all-False 'name available' column.")
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, encoding="latin1")
    df = df[["CWNS_ID", "FACILITY_NAME"]].dropna().drop_duplicates(subset="CWNS_ID")
    out = dict(zip(df["CWNS_ID"].astype(str), df["FACILITY_NAME"].astype(str)))
    _CACHE[key] = out
    return out


def facility_idf(names: dict[str, str]) -> tuple[dict[str, float], float]:
    """IDF of distinctive tokens across ALL facility names. Returns (idf,
    default) where default is the weight for a token seen in no name."""
    key = ("idf", id(names))
    if key in _CACHE:
        return _CACHE[key]
    df_count: dict[str, int] = {}
    for n in names.values():
        for t in set(distinctive(normalise(n))):
            df_count[t] = df_count.get(t, 0) + 1
    N = max(len(names), 1)
    idf = {t: math.log((1 + N) / (1 + c)) + 1.0 for t, c in df_count.items()}
    default = math.log(1 + N) + 1.0
    _CACHE[key] = (idf, default)
    return idf, default


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def add_name_features(df: pd.DataFrame, names: dict[str, str] | None = None,
                      id_col: str = "CWNS_ID", owner_col: str = "owner") -> pd.DataFrame:
    """Adds name_match_score (0..1) and name_match_available (bool).

    Call BEFORE 02's add_name_matching(), which drops the raw owner column.
    Row order and count are preserved.
    """
    from rapidfuzz import fuzz, process

    if names is None:
        names = load_facility_names()
    idf, idf_default = facility_idf(names)

    df = df.copy()
    n = len(df)
    score = np.zeros(n, dtype=float)
    if n == 0 or owner_col not in df.columns:
        df["name_match_score"] = score
        df["name_match_available"] = np.zeros(n, dtype=bool)
        return df

    owners = df[owner_col].fillna("").astype(str).str.strip()
    owner_codes, owner_uniques = pd.factorize(owners)
    owner_norm = [normalise(o) for o in owner_uniques]
    owner_inits = [initials(o) for o in owner_norm]
    owner_raw = [raw_tokens(o) for o in owner_uniques]

    # Inverted index: owner token -> owner ids containing it
    vocab_index: dict[str, list[int]] = {}
    for oid, o in enumerate(owner_norm):
        for t in set(o.split()):
            vocab_index.setdefault(t, []).append(oid)
    vocab = list(vocab_index)
    # Initials-prefix index for acronym lookups (owner initials of length L
    # indexed under every prefix and suffix of length >= 3)
    inits_index: dict[str, set[int]] = {}
    for oid, ini in enumerate(owner_inits):
        if len(ini) < 3:
            continue
        for k in range(3, len(ini) + 1):
            inits_index.setdefault(ini[:k], set()).add(oid)
            inits_index.setdefault(ini[-k:], set()).add(oid)
    # Owner-side acronym tokens: raw 3-7 letter tokens not already a word
    acr_index: dict[str, set[int]] = {}
    for oid, toks in enumerate(owner_raw):
        for t in toks:
            if 3 <= len(t) <= 7 and t.isalpha() and t not in GENERIC:
                acr_index.setdefault(t, set()).add(oid)

    ids = df[id_col].astype(str).to_numpy()
    has_owner = (owners.str.len() > 0).to_numpy()
    has_name = np.array([bool(names.get(i)) for i in ids])

    rows_by_plant = pd.Series(np.arange(n)).groupby(ids).indices
    for cwns, rows in rows_by_plant.items():
        fac = names.get(cwns)
        if not fac:
            continue
        fnorm = normalise(fac)
        ftoks = distinctive(fnorm)
        fraw = [t for t in raw_tokens(fac) if 3 <= len(t) <= 7 and t.isalpha()
                and t not in GENERIC and t not in ABBREV]
        weights = {t: idf.get(t, idf_default) for t in ftoks}
        # raw acronym tokens that normalisation expanded away still carry weight
        for t in fraw:
            weights.setdefault(t, idf.get(t, idf_default))
        total = sum(weights.values())
        if total <= 0:
            continue

        plant_owners = set(owner_codes[rows])
        best: dict[str, dict[int, float]] = {t: {} for t in weights}
        for t in weights:
            hits = process.extract(t, vocab, scorer=fuzz.ratio,
                                   score_cutoff=FUZZY_CUTOFF, limit=None)
            for tok, s, _ in hits:
                for oid in vocab_index[tok]:
                    if oid in plant_owners and s > best[t].get(oid, 0):
                        best[t][oid] = s
            # facility token is an acronym of the owner's words
            if 3 <= len(t) <= 7 and t.isalpha():
                for oid in inits_index.get(t, ()):
                    if oid in plant_owners and acronym_hit(t, owner_inits[oid]):
                        best[t][oid] = 100.0
        # owner token is an acronym of the facility name's words
        finit = initials(fnorm)
        rev_hits: set[int] = set()
        if len(finit) >= 3:
            for k in range(3, len(finit) + 1):
                for key in (finit[:k], finit[-k:]):
                    for oid in acr_index.get(key, ()):
                        if oid in plant_owners:
                            rev_hits.add(oid)

        owner_score: dict[int, float] = {}
        for t, per in best.items():
            w = weights[t]
            for oid, s in per.items():
                owner_score[oid] = owner_score.get(oid, 0.0) + w * s / 100.0
        for oid in owner_score:
            owner_score[oid] /= total
        for oid in rev_hits:
            owner_score[oid] = 1.0
        if owner_score:
            codes = owner_codes[rows]
            score[rows] = [owner_score.get(c, 0.0) for c in codes]

    available = has_owner & has_name
    score[~available] = 0.0
    df["name_match_score"] = np.clip(score, 0.0, 1.0)
    df["name_match_available"] = available
    return df


def add_pool_features(df: pd.DataFrame, prefix: str, id_col: str = "CWNS_ID",
                      score_col: str = "name_match_score") -> pd.DataFrame:
    """Pool context for the name score, within each plant's rows of df:
      {prefix}_rank      1 = best score in the pool (ties share the rank)
      {prefix}_gap       pool best minus this candidate's score
      {prefix}_n_at_max  candidates sharing the pool's best score (0 if the
                         best is 0) -- 'the city owns six parcels here'
      {prefix}_is_max    this candidate holds the best, non-zero score
    Computed over whatever rows are passed: the full ring for Stage 2a,
    the top-20 for the re-ranker. Same arithmetic in both places."""
    df = df.copy()
    s = df[score_col].astype(float)
    g = s.groupby(df[id_col])
    pmax = g.transform("max")
    df[f"{prefix}_rank"] = g.rank(method="min", ascending=False).astype(int)
    df[f"{prefix}_gap"] = (pmax - s).to_numpy()
    at_max = (s == pmax) & (pmax > 0)
    df[f"{prefix}_n_at_max"] = at_max.groupby(df[id_col]).transform("sum").astype(int)
    df[f"{prefix}_is_max"] = at_max.to_numpy()
    return df
