"""
inspect_deployed_02.py
========================
Prints the actual deployed build_discharge_features() function (and any
other FACILITY_ID references) from the running copy of
02_feature_engineering.py, to check whether it matches what's expected --
without needing shell grep access.

Usage:
    python inspect_deployed_02.py
"""
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent / "02_feature_engineering.py"


def main():
    print(f"Reading: {SCRIPT_PATH}\n")
    lines = SCRIPT_PATH.read_text().splitlines()

    print("--- Every line mentioning FACILITY_ID or DISCHARGE_TYPE ---")
    for i, line in enumerate(lines, start=1):
        if "FACILITY_ID" in line or "DISCHARGE_TYPE" in line:
            print(f"  {i}: {line}")

    print("\n--- Full build_discharge_features function ---")
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def build_discharge_features"):
            start = i
            break
    if start is None:
        print("  FUNCTION NOT FOUND -- this is itself informative")
    else:
        end = start + 1
        while end < len(lines) and not (lines[end].startswith("def ") and end > start + 1):
            end += 1
        for i in range(start, min(end, start + 40)):
            print(f"  {i+1}: {lines[i]}")

    print("\n--- Every line mentioning STATE_CODE ---")
    for i, line in enumerate(lines, start=1):
        if "STATE_CODE" in line:
            print(f"  {i}: {line}")


if __name__ == "__main__":
    main()
