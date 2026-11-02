# Notes on the first full run

`calls.csv` and `turns.csv` hold 880 calls: 4 speech-to-text setups, 4 audio conditions and
22 scenarios (3 noise seeds per noisy condition), plus 22 calls with exact text as a baseline.

Call pass rate (%):

| Setup | Clean | Phone | Phone + street 10 dB | Phone + crowd 5 dB |
|---|---|---|---|---|
| tiny | 4.5 | 6.1 | 13.6 | 16.7 |
| small | 31.8 | 28.8 | 19.7 | 10.6 |
| medium | 63.6 | 51.5 | 39.4 | 25.8 |
| small+hint | 22.7 | 30.3 | 22.7 | 21.2 |

No call ended in a wrong action. Every failure was a handoff to a human.

## Read these numbers with care

These are simulated calls with synthesised voices, and the agent is a simple rule-based baseline.
Pass rates describe this setup only, not Whisper or voice agents in general.

1. **The `cause` column overstates speech errors.** It says `speech` whenever the agent passes on
   exact text. Many of those failures are the agent's strict word matching: "दीजिये" against
   "दीजिए", or "कानसल" against "कैंसल". Others are real mishearing, such as "कर" for "कल".
2. **Clean and phone conditions are underrated compared with noisy ones.** A caller who repeats
   a line sends the same clip again, so a retry barely changes the transcript under a model like
   medium (0% changed on clean, 100% under crowd noise). Noise gives each retry a fresh attempt.
   This likely explains why `tiny` appears to improve as noise increases: 8 of its 11 crowd passes
   needed a retry. It has not been confirmed by re-running with varied retries.
3. **Crowd noise contains the caller's own lines.** Removing them barely changed `tiny` in a
   separate test (9 of 66 passes with them, 8 of 66 without), so this is not the main effect.
4. **Repeat runs differ.** The same `tiny` crowd condition gave 11 of 66 here and 9 of 66 in that
   test, so differences of a few calls are not meaningful.
5. **`small+hint` is not a fair win.** The hint names words the scenarios use, such as "कल" and
   "परसों", and Whisper sometimes echoes it back ("पुष्टि।"): 54 of 795 turns, including 13% of
   turns under crowd noise, and 4 of its 54 passes.
6. **Timing is unreliable.** The laptop went into standby during the run (one turn shows 31,323 s),
   other stalls reached 320 s, and the power cap changed. Do not quote the p95 or maximum times.
