"""
facilities.py
=============
CWNS facility names, from a local flat file separate from everything else
this app reads (not Regrid, not synced from HPC). Loaded once into memory on
first use -- 30,881 rows is small enough that a per-request lookup would be
needless overhead, unlike the Regrid store which genuinely needs DuckDB.

FACILITY_NAME was the one piece of "context" flagged as unverified when this
app was first built -- no CWNS table seen at the time had an explicit name
field. This file has it. OWNER_TYPE (Public/Private) is included too since
it's free from the same read and directly useful context, even though it
wasn't explicitly asked for -- everything else in the file (DESCRIPTION,
SUPERFUND_FLAG, etc.) is left alone rather than scope-creeping this.
"""
import pandas as pd

import config as C

_facility_lookup = None


def _load():
    global _facility_lookup
    if _facility_lookup is not None:
        return _facility_lookup

    if not C.FACILITIES_PATH.exists():
        print(f"  WARNING: {C.FACILITIES_PATH} not found -- facility names "
              f"will be unavailable. Check config.py's FACILITIES_PATH.")
        _facility_lookup = {}
        return _facility_lookup

    try:
        df = pd.read_csv(C.FACILITIES_PATH, dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        # Every other CWNS text file this pipeline reads uses latin1
        # (see 02_feature_engineering.py) -- fall back to it if utf-8 fails,
        # rather than assuming this file is the one exception.
        df = pd.read_csv(C.FACILITIES_PATH, dtype=str, encoding="latin1")

    df = df[["CWNS_ID", "FACILITY_NAME", "OWNER_TYPE"]].drop_duplicates(subset="CWNS_ID")
    _facility_lookup = df.set_index("CWNS_ID")[["FACILITY_NAME", "OWNER_TYPE"]].to_dict("index")
    print(f"  Loaded {len(_facility_lookup)} facility name(s) from {C.FACILITIES_PATH.name}")
    return _facility_lookup


def get_facility_info(cwns_id: str) -> dict:
    """Returns {"facility_name": str|None, "owner_type": str|None}. Never
    raises on a missing CWNS_ID -- just returns Nones, same pattern as the
    parcel lookups."""
    lookup = _load()
    entry = lookup.get(str(cwns_id), {})
    return {
        "facility_name": entry.get("FACILITY_NAME"),
        "owner_type": entry.get("OWNER_TYPE"),
    }
