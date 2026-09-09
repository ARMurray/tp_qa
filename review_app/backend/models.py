"""
models.py
=========
Request/response schemas. VerdictSubmit is the important one -- it's the
concrete implementation of REVIEW_LOOP_PLAN.md Phase 4's four requirements.
"""
from typing import Literal, Optional

from pydantic import BaseModel, model_validator

PlantVerdict = Literal[
    "reported_correct",           # confirm_reported path, or candidate_pick
                                   # reviewer decides the ORIGINAL reported
                                   # point was right after all
    "candidate_correct",          # reviewer picked one of the shown candidates
    "truth_outside_candidates",   # Phase 4 requirement #1 -- the new path.
                                   # Truth is neither the reported point nor
                                   # any shown candidate.
    "needs_info",                 # reviewer can't determine truth right now
]

ConfirmationType = Literal["independent", "confirmed_proposal"]


class VerdictSubmit(BaseModel):
    cwns_id: str
    plant_verdict: PlantVerdict
    reviewer: str

    # required iff plant_verdict == "candidate_correct"
    selected_ll_uuid: Optional[str] = None
    candidate_rank: Optional[int] = None

    # required iff plant_verdict == "truth_outside_candidates"
    truth_latitude: Optional[float] = None
    truth_longitude: Optional[float] = None

    confirmation_type: Optional[ConfirmationType] = None
    reviewer_notes: Optional[str] = None

    @model_validator(mode="after")
    def check_required_fields(self):
        if self.plant_verdict == "candidate_correct":
            if self.selected_ll_uuid is None or self.candidate_rank is None:
                raise ValueError(
                    "candidate_correct requires selected_ll_uuid and candidate_rank")
        if self.plant_verdict == "truth_outside_candidates":
            if self.truth_latitude is None or self.truth_longitude is None:
                raise ValueError(
                    "truth_outside_candidates requires truth_latitude and truth_longitude")
        # Phase 4 #4: confirmation_type only makes sense when the reviewer
        # actually confirmed something -- reported_correct or candidate_correct.
        # Not required for truth_outside_candidates (nothing proposed was
        # confirmed) or needs_info (nothing was decided at all).
        if self.plant_verdict in ("reported_correct", "candidate_correct"):
            if self.confirmation_type is None:
                raise ValueError(
                    f"{self.plant_verdict} requires confirmation_type "
                    f"('independent' or 'confirmed_proposal')")
        return self
