"""
routes/plants.py
=================
GET routes: queue status, nav list, and the next/specific plant with
everything the frontend needs -- plant info, candidates (if any), and
LIVE parcel context (geometry + owner/LBCS/acreage) for the reported point
and every shown candidate, via a single batched DuckDB lookup.
"""
from fastapi import APIRouter, HTTPException

from backend import db, parcels, facilities

router = APIRouter()


@router.get("/status")
def queue_status():
    with db.get_conn() as conn:
        slices = db.counts_by_slice(conn)
    total = sum(s["total"] for s in slices)
    reviewed = sum(s["reviewed"] for s in slices)
    return {
        "total": total,
        "reviewed": reviewed,
        "remaining": total - reviewed,
        "by_slice": slices,
    }


@router.get("/list")
def list_plants(review_round: int = 1):
    """Full plant list for the round, for the navigation panel -- includes
    reviewed status so already-done plants can be badged distinctly, plus
    facility name (cheap: in-memory dict lookup, not a per-row DB query)."""
    with db.get_conn() as conn:
        plants = db.list_all_plants(conn, review_round)
    for p in plants:
        p["facility_name"] = facilities.get_facility_info(p["cwns_id"])["facility_name"]
    return plants


def _load_plant_with_context(plant: dict, candidates: list[dict]) -> dict:
    """Shared by /next and /{cwns_id} -- batches ALL parcel lookups this page
    needs (reported point + every candidate) into one DuckDB query rather
    than one per parcel, plus the facility name/owner-type lookup."""
    state = plant["state_code"]
    uuids_needed = [c["ll_uuid"] for c in candidates]
    if plant["reported_ll_uuid"]:
        uuids_needed.append(plant["reported_ll_uuid"])

    context = parcels.get_parcel_context(state, uuids_needed)
    facility_info = facilities.get_facility_info(plant["cwns_id"])
    plant = {**plant, **facility_info}

    for c in candidates:
        entry = context.get(c["ll_uuid"], {})
        c["geometry"] = entry.get("geometry")
        for col in parcels.ATTR_COLS:
            c[col] = entry.get(col)

    reported_entry = context.get(plant["reported_ll_uuid"], {}) if plant["reported_ll_uuid"] else {}
    reported_geometry = reported_entry.get("geometry")
    reported_context = {col: reported_entry.get(col) for col in parcels.ATTR_COLS}

    return {
        "plant": plant,
        "candidates": candidates,
        "reported_geometry": reported_geometry,
        "reported_context": reported_context,
    }


@router.get("/next")
def next_plant():
    with db.get_conn() as conn:
        plant = db.next_unreviewed(conn)
        if plant is None:
            return {"done": True}
        candidates = db.get_candidates(conn, plant["cwns_id"]) if plant["review_task"] == "candidate_pick" else []

    result = _load_plant_with_context(plant, candidates)
    result["done"] = False
    return result


@router.get("/{cwns_id}")
def get_plant_by_id(cwns_id: str):
    """Look up a specific plant -- used for the nav list, or re-opening a
    plant to double check a submitted verdict."""
    with db.get_conn() as conn:
        plant = db.get_plant(conn, cwns_id)
        if plant is None:
            raise HTTPException(404, f"No plant with CWNS_ID {cwns_id} in the loaded queue")
        candidates = db.get_candidates(conn, cwns_id) if plant["review_task"] == "candidate_pick" else []

    return _load_plant_with_context(plant, candidates)
