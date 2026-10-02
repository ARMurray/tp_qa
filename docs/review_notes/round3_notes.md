# Round 3 review notes

Notes the reviewer sent during the round 3 session (queue built 2026-09-25,
150 plants: 105 uncertain, 45 random). Logged verbatim as they arrive, with
the CWNS_ID where one was given, for analysis after the session.

Tags, added at logging time to make grouping easier later. They are a first
guess, not a conclusion:
`ranking`, `candidates-missing`, `detector`, `stage1-false-flag`,
`reported-correct`, `stage1-threshold`, `queue-composition`, `parcel-data`,
`imagery`, `app-ui`, `data-display`, `other`.

| # | time | CWNS_ID | note (verbatim) | tags |
|---|------|---------|-----------------|------|
| 1 | 09:14 | 17003215001 | the correct candidate is #3. Its the second closest candidate and it has 'sewage' in the owner name. This should be low hanging fruit but it scored 3rd. | ranking |
| 2 | 09:18 | 18005995011 | got 0.503 stage 1 score but its a sliver parcel on undeveloped land. No buildings or anything. Suggest raising the threshold to prevent stage 2 inclusion. | stage1-threshold, parcel-data |
| 3 | 09:27 | — | There are a large number of plants that end up in review that serve less than 500 people. In the future, we need a cross section of plant sizes in our review queue | queue-composition |
| 4 | 09:34 | 20001601001 | candidate 1 was correct. Noting this example because there is another real treatment plant in town that came through as candidate #2 and also had object detections. Model worked perfectly in this case | ranking, detector, success |
| 5 | 09:41 | 26002106001 | stage 1 was 0.983 but it's wrong.  Treatment plant is very close but with no stage 2 run, we can't compare. Wondering if everything should go to stage 2... | stage1-false-pass, pipeline-design |
| 6 | 09:48 | 36000000395 | another case where the correct location is a lower candidate. In this case it's candidate #4 and the parcel owner is 'Suffolk county sewer district NO 2Tallmadge woods' against a treatment plant name of 'Suffolk (Co) SCSD #2 Tallmadge Woods STP'. That should be a slam dunk. Something is missing in our feature engineering, or perhaps we need to be more deliberate in our training testing splits. | ranking, feature-engineering |
| 7 | 09:55 | — | A bunch of treatment plants have 0 population. The cwns is all about future planning so done treatment plants don't even exist yet. Suggest filtering to populations >0 or maybe even 1,000 | queue-composition, universe-filter |
| 8 | 10:02 | 39003892001 | the correct candidate is #2, which is 278m from the reported location and has an object detection and the owner name has Jamestown in it, which is also in the name of the treatment plant. The #1 candidate had a distance of 1549m and no obvious name similarity | ranking, feature-engineering, detector |

---

## Analysis (2026-09-25, from app.db after the session)

**Headline.** Among 87 decided `candidate_pick` plants (45 `needs_info` excluded),
the #1 candidate was right 41 times (**47.1%**, up from 33.6% and 37.6% in
rounds 1 and 2). The truth was somewhere in the top 5 for 74.7%. Caveat: 41 of
the 65 correct picks were `confirmed_proposal`.

**Precision of "move to #1" by re-ranker score** (95% Wilson interval):

| score ≥ | plants | right | precision | CI |
|---|---|---|---|---|
| 0.90 | 13 | 11 | 84.6% | 58–96% |
| 0.85 | 23 | 19 | 82.6% | 63–93% |
| 0.80 | 31 | 24 | 77.4% | 60–89% |
| 0.70 | 45 | 28 | 62.2% | 48–75% |
| all  | 87 | 41 | 47.1% | 37–58% |

No cutoff reaches 90% yet, and the top buckets are too small to prove it if one did.

**Detections.** 22 shown candidates fired: 11 on the true parcel, 11 elsewhere.
The detector fired on only 11 of the 65 true parcels shown (17%), with max
confidences 0.27–0.45. When #1 fired it was right 7 of 9 times.

**Stage 1 passes** (18 `confirm_reported`, all random slice): 15 decided, 11
correct, **4 wrong (27%)**, at Stage 1 scores 0.503, 0.820, 0.983, 0.995. No
cutoff separates them.

**Plant size.** All 8 zero-population plants came back `needs_info`.
`needs_info` by population served: 1–499 43%, 500–999 38%, 1k–10k 19%, 10k+ 0%.

**Notes 1, 6, 8.** The signals the reviewer used (wastewater keyword, plant
name in owner, detection, distance) are present but carry little weight. In
36000000395 the re-ranker demoted the true parcel from Stage 2a's #2 to #4.
The plant-name to owner match (note 6) is not a feature at all.
