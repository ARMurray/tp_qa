"""
Build the two address tables for a batch geocoder, in one layout:
  correction/data/geocode/cwns_addresses_to_geocode.csv
  correction/data/geocode/icis_npdes_addresses_to_geocode.csv

First five columns are the usual batch-geocoder input (ID, STREET, CITY, STATE, ZIP);
the rest are hints. Nothing is dropped, rough addresses included: the correction model
only needs a location within ~10 km to compare against the CWNS point.
  STREET_RAW      the street line we chose, as written
  ADDRESS_CLASS   what STREET_RAW holds (street_number, road_no_number, intersection,
                  descriptive, plss, po_box, county_only, missing)
  STREET          STREET_RAW when a geocoder can use it (the first three classes), else
                  blank so the geocoder matches on city / ZIP instead of descriptive text
  EXPECTED_PRECISION  address | street | locality | none  (none = no street, city or ZIP)
  GEOCODE_READY   true unless EXPECTED_PRECISION is none

CWNS source: V2 DuckDB, staged.location for every non-planned treatment plant
  (CWNS PHYSICAL_LOCATION, with the 2024 contractor-updated rows for confirmed plants).
  The street is taken from ADDRESS, or from ADDRESS_2 when ADDRESS is empty or is not
  a street with a number. The other line goes to STREET_2, a PO box to PO_BOX.
ICIS source: correction/data/icis_npdes/icis_npdes_permits.csv (get_icis_npdes.R).

Run:  python -I build_geocode_tables.py [path/to/sewersheds_v2.duckdb]
"""
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "correction/data/geocode"
ICIS = REPO / "correction/data/icis_npdes/icis_npdes_permits.csv"
DB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Github/sewersheds/V2/data/db/sewersheds_v2.duckdb")

PLACEHOLDER = {"NULL", "NONE", "N/A", "NA", "UNKNOWN", "UNK", "TBD", "-", ".", "0", "?"}
PO_BOX = re.compile(r"\b(?:P\.? ?O\.? ?BOX|POST OFFICE BOX|BOX [0-9]+)\b")
COUNTY_ONLY = re.compile(r"^(?:IN |IN THE )?[A-Z .'\-]+ (?:COUNTY|PARISH)$")
PLSS = re.compile(r"\bSEC(?:TION|\.)? ?[0-9]|\bT[0-9]+[NS]\b|\bR[0-9]+[EW]\b|\bTWNSH|\bTWP\b|\bRNG\b|\b[NS][EW]Q\b|\bQUARTER\b")
INTERSECTION = re.compile(r"(?:&|\bAND\b|\bINT\.? |\bINTERSECTION\b| AT )")
DESCRIPTIVE = re.compile(r"\b(?:MI|MILES?|FEET|FT|ADJ|ADJACENT|NEAR|BEHIND|NORTH OF|SOUTH OF|EAST OF|WEST OF|OFF OF)\b|\b[NSEW]{1,2} OF\b")
CIVIC = re.compile(r"\b(?:CITY|TOWN|VILLAGE|BOROUGH|TOWNSHIP|COUNTY) (?:HALL|OFFICE|BUILDING)\b"
                   r"|\bMUNICIPAL (?:BUILDING|CENTER|COMPLEX)\b|\bCOURTHOUSE\b")
STREET_NUMBER = re.compile(r"^[0-9]+[A-Z]?(?:[-/][0-9A-Z]+)?\s+(?:[NSEW]{1,2}\.?\s+)?[A-Z0-9]")


def clean(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return "" if s.upper() in PLACEHOLDER else s


def classify(s):
    """What kind of text is this address line?"""
    if s == "":
        return "missing"
    u = s.upper()
    if PO_BOX.search(u):
        return "po_box"
    if COUNTY_ONLY.match(u):
        return "county_only"
    if PLSS.search(u):
        return "plss"
    if DESCRIPTIVE.search(u):
        return "descriptive"
    if INTERSECTION.search(u):
        return "intersection"
    if STREET_NUMBER.match(u):
        return "street_number"
    return "road_no_number"


def pick_street(a1, a2):
    """Choose the line a street geocoder should get; return (street, other_line, po_box)."""
    c1, c2 = classify(a1), classify(a2)
    po = a1 if c1 == "po_box" else (a2 if c2 == "po_box" else "")
    lines = [(a1, c1), (a2, c2)]
    if a2.upper() == a1.upper():
        lines = [(a1, c1)]
    for want in ("street_number", "intersection", "road_no_number", "descriptive", "plss"):
        for text, cls in lines:
            if cls == want:
                other = next((t for t, c in lines if t != text and c not in ("po_box", "county_only", "missing")), "")
                return text, other, po
    return "", "", po


def zip5(z):
    z = re.sub(r"[^0-9]", "", str(z))
    return z[:5] if len(z) >= 5 else ""


def finish(df):
    df["ADDRESS_CLASS"] = df.STREET.map(classify)
    df["FLAG_CIVIC_BUILDING"] = (df.STREET + " " + df.STREET_2).str.upper().str.contains(CIVIC)
    df["FLAG_NO_ZIP"] = df.ZIP == ""
    df["FLAG_NO_CITY"] = df.CITY == ""
    df["STREET_RAW"] = df.STREET
    df["STREET"] = df.STREET_RAW.where(df.ADDRESS_CLASS.isin(["street_number", "road_no_number", "intersection"]), "")
    has_locality = (df.CITY != "") | (df.ZIP != "")
    df["EXPECTED_PRECISION"] = "none"
    df.loc[has_locality, "EXPECTED_PRECISION"] = "locality"
    df.loc[has_locality & df.ADDRESS_CLASS.isin(["road_no_number", "intersection"]), "EXPECTED_PRECISION"] = "street"
    df.loc[has_locality & (df.ADDRESS_CLASS == "street_number"), "EXPECTED_PRECISION"] = "address"
    df["GEOCODE_READY"] = df.EXPECTED_PRECISION != "none"
    return df


def build_rows(df, a1_col, a2_col):
    sel = [pick_street(clean(a), clean(b)) for a, b in zip(df[a1_col], df[a2_col])]
    return pd.DataFrame(sel, columns=["STREET", "STREET_2", "PO_BOX"], index=df.index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB), read_only=True)
    c = con.execute("""SELECT t.cwns_id AS ID, t.facility_name AS NAME, t.in_universe AS IN_UNIVERSE,
                              l.address AS A1, l.address_2 AS A2, l.city AS CITY, t.state_code AS STATE, l.zip5 AS ZIP,
                              l.record_source AS RECORD_SOURCE
                       FROM derived.treatment_plant t JOIN staged.location l USING (cwns_id)
                       WHERE NOT t.planned_new ORDER BY t.cwns_id""").df()
    con.close()
    for col in ("NAME", "CITY", "STATE"):
        c[col] = c[col].map(clean)
    c["ZIP"] = c["ZIP"].map(zip5)
    c = pd.concat([c, build_rows(c, "A1", "A2")], axis=1)
    cw = finish(c[["ID", "STREET", "CITY", "STATE", "ZIP", "STREET_2", "PO_BOX", "NAME", "IN_UNIVERSE", "RECORD_SOURCE"]].copy())
    cw.to_csv(OUT / "cwns_addresses_to_geocode.csv", index=False)

    p = pd.read_csv(ICIS, dtype=str, keep_default_na=False)
    for col in ("CITY", "STATE_CODE", "FACILITY_NAME"):
        p[col] = p[col].map(clean)
    p = pd.concat([p, build_rows(p, "LOCATION_ADDRESS", "SUPPLEMENTAL_ADDRESS_TEXT")], axis=1)
    ic = pd.DataFrame({"ID": p.EXTERNAL_PERMIT_NMBR, "STREET": p.STREET, "CITY": p.CITY, "STATE": p.STATE_CODE,
                       "ZIP": p.ZIP.map(zip5), "STREET_2": p.STREET_2, "PO_BOX": p.PO_BOX, "NAME": p.FACILITY_NAME,
                       "TIER": p.tier, "PERMIT_TYPE": p.PERMIT_TYPE_CODE})
    ic = finish(ic)
    ic["ADDRESS_KEY"] = ic.STREET.str.upper() + "|" + ic.CITY.str.upper() + "|" + ic.STATE + "|" + ic.ZIP
    ic.to_csv(OUT / "icis_npdes_addresses_to_geocode.csv", index=False)

    for name, d in (("CWNS", cw), ("ICIS", ic)):
        print(f"\n{name}: {len(d)} rows, unique ids {d.ID.nunique()}, ready {int(d.GEOCODE_READY.sum())}, "
              f"no_zip {int(d.FLAG_NO_ZIP.sum())}, no_city {int(d.FLAG_NO_CITY.sum())}, with_po_box {int((d.PO_BOX != '').sum())}")
        print(d.EXPECTED_PRECISION.value_counts().to_string())
    print("\nICIS unique address keys among ready rows:", ic[ic.GEOCODE_READY].ADDRESS_KEY.nunique())


if __name__ == "__main__":
    main()
