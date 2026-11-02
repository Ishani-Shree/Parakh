"""Summarise saved runs and draw the report figures.

Usage:
    python -m parakh.report      # writes results/fig_*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CONDITIONS = ["clean", "phone", "phone_street_10db", "phone_crowd_5db"]
CONDITION_LABELS = ["Clean", "Phone line", "Phone + street", "Phone + crowd"]
SETUPS = ["tiny", "small", "medium", "medium+hint"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order; assigned, never cycled
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": TEXT, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def load_calls(folder: Path) -> pd.DataFrame:
    """Calls from a run, without the exact-text baseline rows."""
    calls = pd.read_csv(folder / "calls.csv")
    return calls[calls["config"] != "oracle"]


def load_turns(folder: Path) -> pd.DataFrame:
    turns = pd.read_csv(folder / "turns.csv")
    turns["heard"] = turns["heard"].fillna("")
    return turns[turns["config"] != "oracle"]


def pass_rates(calls: pd.DataFrame) -> pd.DataFrame:
    """Percentage of calls passed: one row per setup, one column per condition, mildest first."""
    table = calls.pivot_table(index="config", columns="condition", values="passed", aggfunc="mean") * 100
    return table.reindex(columns=CONDITIONS)


def condition_counts(calls: pd.DataFrame) -> pd.DataFrame:
    grouped = calls.groupby("condition")["passed"]
    return pd.DataFrame({"passed": grouped.sum().astype(int), "calls": grouped.size()}).reindex(CONDITIONS)


def speed_table(turns: pd.DataFrame) -> pd.DataFrame:
    seconds = turns.groupby("config")["stt_s"]
    return pd.DataFrame({"median_s": seconds.median(), "p95_s": seconds.quantile(0.95), "max_s": seconds.max()})


def plot_held_out(before: pd.DataFrame, after: pd.DataFrame) -> plt.Figure:
    old, new = condition_counts(before), condition_counts(after)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    width = 0.38
    for offset, counts, label, color in ((-width / 2, old, "Old agent", SERIES[0]), (width / 2, new, "New agent", SERIES[1])):
        rates = 100 * counts["passed"] / counts["calls"]
        bars = ax.bar([i + offset for i in range(len(CONDITIONS))], rates, width, label=label, color=color,
                      edgecolor=SURFACE, linewidth=2)
        for bar, n, total in zip(bars, counts["passed"], counts["calls"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{n}/{total}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(CONDITIONS)), CONDITION_LABELS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Calls passed (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("Held-out scenarios: new agent vs old, medium model", loc="left", fontweight="bold")
    return fig


def plot_setups(calls: pd.DataFrame) -> plt.Figure:
    rates = pass_rates(calls)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for setup, color in zip(SETUPS, SERIES):
        if setup not in rates.index:
            continue
        ax.plot(CONDITION_LABELS, rates.loc[setup], color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=setup)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Calls passed (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=2)
    ax.set_title("Main scenarios: pass rate by speech-to-text setup", loc="left", fontweight="bold")
    return fig


def plot_speed(turns: pd.DataFrame) -> plt.Figure:
    speed = speed_table(turns).reindex([s for s in SETUPS if s in set(turns["config"])])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    height = 0.38
    positions = range(len(speed))
    for offset, column, label, color in ((-height / 2, "median_s", "Median", SERIES[0]), (height / 2, "p95_s", "Slowest 5% start at", SERIES[1])):
        bars = ax.barh([p + offset for p in positions], speed[column], height, label=label, color=color,
                       edgecolor=SURFACE, linewidth=2)
        for bar, value in zip(bars, speed[column]):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"{value:.2f} s", va="center", fontsize=8.5)
    ax.set_yticks(list(positions), speed.index)
    ax.invert_yaxis()
    ax.set_xlim(0, speed["p95_s"].max() * 1.15)  # room for the value labels
    ax.set_xlabel("Transcription time per turn (s)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("Most turns are quick, but the slow ones would stall a live call", loc="left", fontweight="bold")
    return fig


def main():
    folder = RESULTS / "phase4"
    calls, turns = load_calls(folder), load_turns(folder)
    before = pd.read_csv(folder / "holdout_before_old_agent.csv")
    after = load_calls(RESULTS / "phase4_holdout")
    for name, fig in (("fig_held_out.png", plot_held_out(before, after)),
                      ("fig_setups.png", plot_setups(calls)),
                      ("fig_speed.png", plot_speed(turns))):
        fig.savefig(RESULTS / name)
        plt.close(fig)
        print("wrote", RESULTS / name)


if __name__ == "__main__":
    main()
