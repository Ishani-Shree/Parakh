"""Run every scenario through every speech-to-text config and noise condition.

Usage:
    python -m parakh.run --oracle-only              # agent logic only, no audio
    python -m parakh.run --models small             # one model
    python -m parakh.run --models tiny small medium small+hint

"oracle" feeds the agent the caller's exact words. If a call passes with the
oracle but fails with real speech, the speech layer caused the failure; if it
fails with the oracle too, the agent itself is at fault.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.io import wavfile

from .agent import RuleBasedCODAgent

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MAX_TURNS = 12


def load_scenarios(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [{**data["latin_defaults" if s["voice"].startswith("en-") else "defaults"], **s}
            for s in data["scenarios"]]


def caller_line(scenario: dict, turn, asked: dict) -> str:
    """What this caller says in response to the agent's current question."""
    if turn.state == "readback":
        right = turn.readback == scenario["expected"].get("pincode")
        return scenario["readback_yes"] if right else scenario["readback_no"]
    line = scenario["replies"].get(turn.state, scenario["fallback"])
    if isinstance(line, list):
        index = min(asked.get(turn.state, 0), len(line) - 1)
        asked[turn.state] = asked.get(turn.state, 0) + 1
        return line[index]
    return line


def all_lines(scenarios: list[dict]) -> list[tuple[str, str]]:
    items = []
    for s in scenarios:
        lines = [s["readback_yes"], s["readback_no"], s["fallback"]]
        for reply in s["replies"].values():
            lines.extend(reply if isinstance(reply, list) else [reply])
        items.extend((line, s["voice"]) for line in lines)
    return items


def score(scenario: dict, result) -> tuple[bool, str]:
    """Return (passed, verdict). A wrong action is worse than handing off to a human."""
    expected = scenario["expected"]
    if result.outcome is None:
        return False, "stuck"
    if result.outcome == "handoff":
        return False, "handoff"
    passed = (result.outcome == expected["outcome"]
              and result.pincode == expected.get("pincode")
              and result.day == expected.get("day"))
    return passed, "pass" if passed else "wrong_action"


def run_call(scenario: dict, hear) -> dict:
    """Play one call. `hear(line, turn_index)` returns (transcript, stt_seconds, audio_seconds)."""
    agent = RuleBasedCODAgent()
    turn = agent.start()
    asked, turns = {}, []
    index = 0
    while turn is not None and index < MAX_TURNS:
        line = caller_line(scenario, turn, asked)
        transcript, stt_s, audio_s = hear(line, index)
        turns.append({"turn": index, "state": turn.state, "agent_said": turn.say, "caller_said": line,
                      "heard": transcript, "stt_s": stt_s, "audio_s": audio_s})
        turn = agent.reply(transcript)
        index += 1
    passed, verdict = score(scenario, agent.result)
    return {"passed": passed, "verdict": verdict, "outcome": agent.result.outcome,
            "pincode": agent.result.pincode, "day": agent.result.day, "turns": turns}


def oracle_hear(line, _index):
    return line, 0.0, 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=str(ROOT / "scenarios" / "cod_confirmation.yaml"))
    ap.add_argument("--models", nargs="*", default=["tiny", "small", "medium"])
    ap.add_argument("--conditions", nargs="*", default=None, help="default: all conditions in audio.py")
    ap.add_argument("--seeds", type=int, default=3, help="noise seeds per scenario")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--oracle-only", action="store_true")
    args = ap.parse_args()

    scenarios = load_scenarios(Path(args.scenarios))
    RESULTS.mkdir(exist_ok=True)
    calls, turns = [], []

    def record(config, condition, seed, scenario, call):
        calls.append({"config": config, "condition": condition, "seed": seed, "scenario": scenario["id"],
                      "tags": ",".join(scenario.get("tags", [])), "passed": call["passed"],
                      "verdict": call["verdict"], "outcome": call["outcome"],
                      "expected_outcome": scenario["expected"]["outcome"], "turns": len(call["turns"])})
        turns.extend({"config": config, "condition": condition, "seed": seed, "scenario": scenario["id"], **t}
                     for t in call["turns"])

    oracle_passed = {}
    for s in scenarios:
        call = run_call(s, oracle_hear)
        oracle_passed[s["id"]] = call["passed"]
        record("oracle", "text", 0, s, call)
    print(f"Oracle (agent logic only): {sum(oracle_passed.values())}/{len(scenarios)} passed")
    for sid, ok in oracle_passed.items():
        if not ok:
            print(f"  agent bug, no audio involved: {sid}")

    if not args.oracle_only:
        from . import audio as audio_lib  # heavy and needs network access; the oracle run does not

        args.conditions = args.conditions or list(audio_lib.CONDITIONS)
        print("Synthesising caller audio...")
        audio_lib.synthesise_all(all_lines(scenarios))
        pool = [audio_lib.load_speech(t, v) for t, v in sorted(set(all_lines(scenarios)))]

        from .stt import Transcriber
        fail_dir = RESULTS / "failures"
        fail_dir.mkdir(exist_ok=True)
        for config in args.models:
            print(f"\nLoading {config}...")
            stt = Transcriber(config, args.device)
            for condition in args.conditions:
                t0 = time.perf_counter()
                seeds = 1 if condition == "clean" else args.seeds  # no noise, so repeats would be identical
                passed = total = 0
                for seed in range(seeds):
                    for s in scenarios:
                        clips = {}

                        def hear(line, index):
                            speech = audio_lib.load_speech(line, s["voice"])
                            noisy = audio_lib.degrade(
                                speech, condition, audio_lib.seed_for(s["id"], index, condition, seed), pool)
                            clips[index] = noisy
                            text, seconds = stt.transcribe(noisy)
                            return text, seconds, len(noisy) / audio_lib.SAMPLE_RATE

                        call = run_call(s, hear)
                        record(config, condition, seed, s, call)
                        passed += call["passed"]
                        total += 1
                        if not call["passed"] and seed == 0:
                            for index, clip in clips.items():
                                name = f"{config}__{condition}__{s['id']}__{index}.wav".replace("+", "_")
                                wavfile.write(fail_dir / name, audio_lib.SAMPLE_RATE,
                                              (np.clip(clip, -1, 1) * 32767).astype(np.int16))
                print(f"  {condition:20s} {passed:3d}/{total:<3d} passed  ({time.perf_counter() - t0:.0f}s)")
            del stt

    calls_df = pd.DataFrame(calls)
    calls_df["oracle_passed"] = calls_df["scenario"].map(oracle_passed)
    calls_df["cause"] = np.where(calls_df["passed"], "", np.where(calls_df["oracle_passed"], "speech", "agent"))
    calls_df.to_csv(RESULTS / "calls.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(turns).to_csv(RESULTS / "turns.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(calls_df)} calls to {RESULTS}")


if __name__ == "__main__":
    main()
