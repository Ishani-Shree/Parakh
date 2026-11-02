import pandas as pd

from parakh.report import CONDITIONS, condition_counts, pass_rates, speed_table


def calls_frame():
    return pd.DataFrame({
        "config": ["medium", "medium", "medium", "tiny"],
        "condition": ["phone", "phone", "clean", "phone"],
        "passed": [True, False, True, False],
    })


def test_pass_rates_are_percentages_in_condition_order():
    table = pass_rates(calls_frame())
    assert list(table.columns) == CONDITIONS
    assert table.loc["medium", "phone"] == 50
    assert table.loc["medium", "clean"] == 100
    assert table.loc["tiny", "phone"] == 0


def test_condition_counts():
    counts = condition_counts(calls_frame())
    assert counts.loc["phone", "passed"] == 1
    assert counts.loc["phone", "calls"] == 3


def test_speed_table():
    turns = pd.DataFrame({"config": ["tiny"] * 3, "stt_s": [1.0, 2.0, 9.0]})
    assert speed_table(turns).loc["tiny", "median_s"] == 2.0
    assert speed_table(turns).loc["tiny", "max_s"] == 9.0
