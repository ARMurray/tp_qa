"""12_score_holdout's multi-source end-to-end outcomes.   python correction/tests/test_holdout_multi.py"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import config as C

spec = importlib.util.spec_from_file_location("s12", HERE.parent / "scripts" / "12_score_holdout.py")
s12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s12)

held = pd.DataFrame({"CWNS_ID": list("ABCDEFG"), "bin": ["corrections"] * 5 + ["correct", "correct"]})
truth = pd.DataFrame({"CWNS_ID": list("ABCDEFG"),
                      "true_ll_uuid": ["P1", "P1", "P1", "P1", None, "P3", "P3"]})
summ = pd.DataFrame({
    "CWNS_ID": list("ABCDEFG"),
    "trigger_reason": ["none", "none", "low_confidence", "low_confidence", "none", "none", "low_confidence"],
    "accepted_site_id": ["A_s1", "B_s1", None, None, "E_s1", "F_s1", None],
    "accepted_sources": ["icis_fac_frs", "cwns_point", None, None, "cwns_point", "cwns_point+icis_fac_frs", None],
    "reported_ll_uuid": ["P1", "P3", "P3", "P3", "P3", "P3", "P3"]})
cand = pd.DataFrame({"CWNS_ID": ["C", "C", "D", "D"], "ll_uuid": ["P1", "P3", "P3", "P9"], "rerank_rank": [1, 2, 1, 2]})
sites = {"A_s1": ["P1", "P2"], "B_s1": ["P3"], "E_s1": ["P3"], "F_s1": ["P3"]}

out = s12.multi_source_outcomes(held, truth, summ, cand, sites).set_index("CWNS_ID")
assert out.at["A", "outcome"] == "accepted_right" and out.at["A", "accepted_sources"] == "icis_fac_frs"   # ICIS site was the truth
assert out.at["B", "outcome"] == "accepted_wrong"                                                          # false accept
assert out.at["C", "outcome"] == "moved_right" and out.at["D", "outcome"] == "flagged_wrong"
assert out.at["E", "outcome"] == "truth_unresolved"
assert out.at["F", "outcome"] == "accepted_right" and out.at["G", "outcome"] == "flagged_wrong"            # a false move
import contextlib, io
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s12.print_multi_source(out.reset_index())
text = buf.getvalue()
print(text)
assert "accepted-WRONG by sources: cwns_point 1" in text, text                         # B's false accept
assert "FALSE-MOVE RATE       1 / 2 = 50.0%" in text, text                             # G flagged_wrong; F accepted_right
# per-plant table: multi-source uses the outcomes as they are; single-seed maps flags to the same labels
band = {"corrections": (pd.Series(list("ABCDE")), pd.Series({"A": True, "C": True, "D": False})),
        "correct": (pd.Series(list("FG")), pd.Series({"F": False, "G": True}))}
pp = s12.per_plant_table(band, summ, truth, out.reset_index()).set_index("CWNS_ID")
assert pp.at["A", "right"] and pp.at["C", "right"] and not pp.at["B", "right"] and not pp.at["G", "right"]
assert pp.at["F", "right"] and not pp.at["E", "scored"] and pp["scored"].sum() == 6
band["corrections"] = (pd.Series(list("ABCDE")), pd.Series({"C": True, "D": False}))   # single-seed: accepted = no #1
pp1 = s12.per_plant_table(band, summ, truth, None).set_index("CWNS_ID")
assert pp1.at["A", "outcome"] == "accepted_reported" and not pp1.at["A", "right"] and pp1.at["C", "right"]
assert pp1.at["B", "outcome"] == "accepted_reported" and pp1.at["C", "outcome"] == "moved_right"
assert pp1.at["D", "outcome"] == "flagged_wrong" and pp1.at["E", "outcome"] == "truth_unresolved"
assert pp1.at["F", "outcome"] == "accepted_right" and pp1.at["G", "outcome"] == "flagged_wrong"
print("per-plant table ok")
print("holdout multi-source tests passed")
