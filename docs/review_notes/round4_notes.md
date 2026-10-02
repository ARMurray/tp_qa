# Round 4 review notes

Notes the reviewer sent during the round 4 session (queue built 2026-10-01,
150 plants: 105 uncertain, 45 random; plants serving > 1,000 people only;
first round from the models with the new detector and name-match features).
Logged verbatim, with the CWNS_ID where one was given, for analysis after
the session.

| # | time | CWNS_ID | note (verbatim) | tags |
|---|------|---------|-----------------|------|
| 1 | 2026-10-01 | — | I see sewer keywords in parcels where stage 1 had a very low score. If a reported location is on a parcel with an owner name that has words like sewer, sewage, sewage, water etc... I would expect a very high stage 1 score, so we should dig into that. | stage1-false-flag, feature-engineering |

---

## Analysis (2026-10-01/02)

**Verdicts:** 150 reviewed -- 75 candidate_correct, 30 reported_correct,
19 truth_outside_candidates, 26 needs_info (17%, down from 32% in round 3).
All 124 decided verdicts folded into the master (`CWNS_Locations_20261001`).

**Precision of "move to #1" (out of sample -- first round these models never saw):**

| re-ranker score >= | plants | right | precision | 95% CI |
|---|---|---|---|---|
| 0.95 | 16 | 15 | 93.8% | 72-99% |
| 0.85 | 30 | 25 | 83.3% | 66-93% |
| 0.80 | 32 | 26 | 81.2% | 65-91% |
| all | 106 | 50 | 47.2% | 38-57% |

Requiring a detection on #1 did not raise precision at the top. Stage 1 flagged
14 plants whose reported location was right; of 15 decided plants it passed, 2
were wrong.

**Note 1 confirmed** (`stage1_keyword_diagnostic`): in Stage 1's labels a
keyword / utility-owned reported parcel is Correct 99.2% of the time (3 of 330
wrong), but without an OSM tag the model scored such parcels at mean 0.70 and
flagged 25% -- it leaned on `osm_ww` (importance 0.34). In this round, 6 of the
14 false flags were this pattern; the 2 genuinely-wrong utility-owned plants
were large regional utilities (Renewable Water Resources, Toho Water Authority).
Response (`db8ddbc`): OSM-tagged reported parcels pass by rule, `osm_ww` dropped
as a Stage 1 feature (rows kept), `utility_owner` feature added. Guardrail held
pending the retrain.

**Queue gap:** 154 of 660 shown candidates had no detection result -- 01e
examined the first build's top-5 and 05b then re-ranked others into view. Next
queue: run 01e on each queued plant's top-20.
