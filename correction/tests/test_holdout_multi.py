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
s12.print_multi_source(out.reset_index())
print("holdout multi-source tests passed")
