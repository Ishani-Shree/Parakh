"""Parakh results dashboard: headline numbers, charts, and failed calls you can listen to.

Run:  streamlit run dashboard.py
"""

import pandas as pd
import streamlit as st

from parakh.report import (CONDITION_LABELS, CONDITIONS, RESULTS, condition_counts, load_calls, load_turns,
                           pass_rates, plot_held_out, plot_setups, plot_speed, speed_table)

st.set_page_config(page_title="Parakh", layout="wide")
st.title("Parakh")
st.caption("Stress-testing a voice agent against difficult Hinglish phone calls. "
           "All calls are simulated, with synthetic voices and simulated noise.")

RUNS = {"Main scenarios": RESULTS / "phase4", "Held-out scenarios": RESULTS / "phase4_holdout"}


@st.cache_data
def load_data():
    return {name: (load_calls(path), load_turns(path)) for name, path in RUNS.items()}


data = load_data()
main_calls, main_turns = data["Main scenarios"]
held_calls, _ = data["Held-out scenarios"]
before = pd.read_csv(RESULTS / "phase4" / "holdout_before_old_agent.csv")
same_retries = load_calls(RESULTS / "phase4_same_retries")

results_tab, failures_tab = st.tabs(["Results", "Listen to failed calls"])

with results_tab:
    new_agent_calls = pd.concat([main_calls, held_calls, same_retries])
    speed = speed_table(main_turns)
    a, b, c, d = st.columns(4)
    a.metric("Held-out pass rate, old agent", f"{before['passed'].mean():.0%}", f"{before['passed'].sum()} of {len(before)} calls",
             delta_color="off")
    b.metric("Held-out pass rate, new agent", f"{held_calls['passed'].mean():.0%}",
             f"{held_calls['passed'].sum()} of {len(held_calls)} calls", delta_color="off")
    c.metric("Wrong actions", int((new_agent_calls["verdict"] == "wrong_action").sum()),
             f"in {len(new_agent_calls):,} calls", delta_color="off")
    d.metric("Medium, slowest 5% of turns", f"{speed.loc['medium', 'p95_s']:.1f} s or more",
             f"median {speed.loc['medium', 'median_s']:.2f} s", delta_color="off")

    left, right = st.columns(2)
    left.pyplot(plot_held_out(before, held_calls))
    right.pyplot(plot_setups(main_calls))
    st.pyplot(plot_speed(main_turns), use_container_width=False)

    with st.expander("Data tables"):
        st.write("Main scenarios: % of calls passed")
        table = pass_rates(main_calls).round(1)
        table.columns = CONDITION_LABELS
        st.dataframe(table)
        st.write("Held-out scenarios: new agent")
        st.dataframe(condition_counts(held_calls))
        st.write("Speed: seconds per turn")
        st.dataframe(speed.round(3))

    st.warning("These results come from simulated calls and a simple rule-based agent. "
               "They do not show how real phone calls or production voice agents perform. "
               "Details are in results/phase4/NOTES.md.")

with failures_tab:
    st.write("Each failed call shows what the caller said, what the speech-to-text heard, and the audio it was given.")
    run = st.selectbox("Scenarios", list(RUNS))
    calls, turns = data[run]
    clips = RUNS[run] / "failures"
    col1, col2, col3 = st.columns(3)
    config = col1.selectbox("Setup", sorted(calls["config"].unique()), index=sorted(calls["config"].unique()).index("medium"))
    condition = col2.selectbox("Condition", CONDITIONS, index=1, format_func=lambda c: CONDITION_LABELS[CONDITIONS.index(c)])
    # Audio is kept for the first noise seed only.
    failed = calls[(calls["config"] == config) & (calls["condition"] == condition) & (calls["seed"] == 0) & ~calls["passed"]]
    if failed.empty:
        st.success("No failed calls for this choice.")
    else:
        scenario = col3.selectbox(f"Failed call ({len(failed)})", failed["scenario"])
        call = failed[failed["scenario"] == scenario].iloc[0]
        st.write(f"Expected: **{call['expected_outcome']}**, result: **{call['outcome']}**, "
                 f"likely cause: **{call['cause']}** (a rough guide, based on spelling similarity)")
        script = turns[(turns["config"] == config) & (turns["condition"] == condition) & (turns["seed"] == 0)
                       & (turns["scenario"] == scenario)].sort_values("turn")
        if not clips.exists():
            st.info("Audio clips are not stored in git. Run the evaluation to create them; transcripts are shown below.")
        for turn in script.itertuples():
            st.markdown(f"**Agent:** {turn.agent_said}  \n**Caller said:** {turn.caller_said}  \n**Heard:** {turn.heard or '(nothing)'}")
            path = clips / f"{config}__{condition}__{scenario}__{turn.turn}.wav".replace("+", "_")
            if path.exists():
                st.audio(str(path))
