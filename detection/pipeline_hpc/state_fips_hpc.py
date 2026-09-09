"""
state_fips.py
=============
Two-digit Census state FIPS code <-> USPS state abbreviation.

CWNS_ID's first two characters are the FIPS code (used as the state-pipeline
job argument and as the filter). Regrid parcel storage is nested by USPS
abbreviation (`state=XX/...`), so 07_run_state_pipeline.py needs to translate
between the two.
"""

FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
    # Territories -- drop from ABBR_TO_FIPS usage if your parcel store /
    # NAIP coverage / CWNS universe doesn't actually include these.
    "60": "AS", "66": "GU", "69": "MP", "72": "PR", "78": "VI",
}
ABBR_TO_FIPS = {v: k for k, v in FIPS_TO_ABBR.items()}

# All 50 states + DC, FIPS order -- used by submit_all_states.sh
CONUS_PLUS_DC_FIPS = [f for f in FIPS_TO_ABBR if f not in ("02", "15", "60", "66", "69", "72", "78")]


def fips_to_abbr(fips: str) -> str:
    fips = str(fips).zfill(2)
    if fips not in FIPS_TO_ABBR:
        raise ValueError(f"Unknown state FIPS code: {fips!r}")
    return FIPS_TO_ABBR[fips]


def abbr_to_fips(abbr: str) -> str:
    abbr = abbr.upper()
    if abbr not in ABBR_TO_FIPS:
        raise ValueError(f"Unknown state abbreviation: {abbr!r}")
    return ABBR_TO_FIPS[abbr]
