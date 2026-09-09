"""
routes/verdicts.py
===================
POST route for submitting a review verdict. Validation of which fields are
required per verdict type lives in models.VerdictSubmit -- this route just
persists an already-validated payload and stamps reviewed_at.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from backend import db
from backend.models import VerdictSubmit

router = APIRouter()


@router.post("")
def submit_verdict(payload: VerdictSubmit):
    with db.get_conn() as conn:
        plant = db.get_plant(conn, payload.cwns_id)
        if plant is None:
            raise HTTPException(404, f"No plant with CWNS_ID {payload.cwns_id} in the loaded queue")
        if plant["reviewed"]:
            raise HTTPException(
                409, f"CWNS_ID {payload.cwns_id} was already reviewed at "
                     f"{plant['reviewed_at']} by {plant['reviewer']}. "
                     f"Re-submission isn't allowed through this route -- if a "
                     f"correction is genuinely needed, that's a deliberate "
                     f"admin action, not a normal review-flow POST.")

        # truth_rank == candidate_rank UNLESS truth is outside the candidate
        # pool entirely -- Phase 4 #2's whole point is that this single field
        # distinguishes "found but mis-ranked" from "not in the pool at all",
        # so it must never silently default to the same value in both cases.
        truth_rank = None
        if payload.plant_verdict == "candidate_correct":
            truth_rank = payload.candidate_rank
        elif payload.plant_verdict == "reported_correct":
            truth_rank = None  # reported point isn't a ranked candidate

        verdict = dict(
            plant_verdict=payload.plant_verdict,
            selected_ll_uuid=payload.selected_ll_uuid,
            candidate_rank=payload.candidate_rank,
            truth_rank=truth_rank,
            truth_latitude=payload.truth_latitude,
            truth_longitude=payload.truth_longitude,
            confirmation_type=payload.confirmation_type,
            reviewer_notes=payload.reviewer_notes,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            reviewer=payload.reviewer,
        )
        db.submit_verdict(conn, payload.cwns_id, verdict)

    return {"ok": True, "cwns_id": payload.cwns_id}
