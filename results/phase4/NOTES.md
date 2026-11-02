# Phase 4 results

Changes since the first run in `results/`: the agent matches sound-alike spellings, retries are
spoken at different speeds, crowd noise excludes the caller's own lines, and the hint is a plain
sentence about the call.

## Result on held-out scenarios

12 scenarios written before the fix and never used to tune it, run with `medium`.

| Condition | Old agent | New agent |
|---|---|---|
| Clean | 5 of 12 | 10 of 12 |
| Phone | 15 of 36 | 31 of 36 |
| Phone + street 10 dB | 15 of 36 | 24 of 36 |
| Phone + crowd 5 dB | 8 of 36 | 15 of 36 |
| **Total** | **43 of 120 (36%)** | **80 of 120 (67%)** |

The old-agent run used identical retries and the new-agent run used varied retries. Stage 3 below
shows that difference is small.

Still weak: `holdout_reschedule_next_week` passed 0 of 10 and `holdout_cancel_found_cheaper` 2 of 10.

## Result on the main scenarios

Passed calls. These scenarios informed the fix, so they may flatter it.

| Setup | Clean (of 22) | Phone (of 66) | Street (of 66) | Crowd (of 66) | Overall |
|---|---|---|---|---|---|
| tiny | 2 | 11 | 14 | 8 | 16% |
| small | 11 | 28 | 22 | 10 | 32% |
| medium | 17 | 48 | 47 | 24 | 62% |
| medium+hint | 13 | 40 | 30 | 16 | 45% |

`medium` in the first run: 14, 34, 26 and 17, or 41% overall.

## What caused the gain

`medium` with identical retries scored 16, 49, 44 and 25 (61% overall), against 62% with varied
retries. Most of the gain from 41% is therefore not from varying retries. The likeliest cause is the
agent fix, but crowd-noise and retry changes were not isolated one at a time.

The new hint scored lower than no hint in every condition, and its words appeared in 24 of 794
transcripts. It should not be used.

## Safety

No wrong actions in 1,220 calls with the new agent. Looser matching did not make it confirm or cancel
wrongly in these tests. One `tiny` call got stuck without an outcome.

## Speed

Per turn, on batch transcription of finished clips, with the GPU cap logged at about 90 W at the start
of the main run:

| Setup | Median | p95 | Max |
|---|---|---|---|
| tiny | 0.27 s | 1.8 s | 2.9 s |
| small | 0.16 s | 5.2 s | 10.9 s |
| medium | 0.33 s | 8.9 s | 12.8 s |

14.4% of `medium` turns took over 2 s. Most turns are quick, but a live call could not absorb the slow
ones. Whisper re-decoding unclear audio is the likely reason, but that has not been tested.
The medians also exclude LLM, speech synthesis and network time.

## Limits

- Voices are synthesised and noise is simulated. None of this is real phone audio, and no regional accents are tested.
- I wrote the held-out scenarios too, so they share my assumptions, and 12 scenarios is a small set.
- Calls from the same scenario are not independent, so 120 calls is not 120 separate tests.
- Barge-in (a caller interrupting the agent) is not tested.
- `close_but_rejected` and `misheard` are a rough split based on spelling similarity.
- The agent is a rule-based baseline, not an LLM or a production agent.
