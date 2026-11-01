from pathlib import Path

import pytest

from parakh.normalize import detect_day, detect_intent, detect_yes_no, extract_digits
from parakh.run import load_scenarios, oracle_hear, run_call

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios" / "cod_confirmation.yaml"


@pytest.mark.parametrize("text, digits", [
    ("five sixty double zero one", "560001"),
    ("four double zero double zero one", "400001"),
    ("five sixty, zero thirty four", "560034"),
    ("पाँच छह शून्य शून्य शून्य एक", "560001"),
    ("पांच छह शून्य शून्य शून्य एक।", "560001"),
    ("तीन zero दो zero one सात", "302017"),
    ("५६०००१", "560001"),
    ("mera pincode 560 001 hai", "560001"),
    ("triple nine", "999"),
    ("no number here", ""),
])
def test_extract_digits(text, digits):
    assert extract_digits(text) == digits


@pytest.mark.parametrize("text, intent", [
    ("हाँ, भेज दीजिए।", "confirm"),
    ("Please cancel कर दो, मत भेजो", "cancel"),
    ("नहीं नहीं, cancel मत करो, भेज दो", "confirm"),
    ("मुझे पता बदलना है", "change_address"),
    ("अभी नहीं, मैं घर पर नहीं हूँ", "reschedule"),
    ("No problem, you can send it", "confirm"),
    ("I know this", None),
])
def test_detect_intent(text, intent):
    assert detect_intent(text) == intent


def test_yes_no_and_day():
    assert detect_yes_no("हाँ, सही है।") == "yes"
    assert detect_yes_no("नहीं, गलत है") == "no"
    assert detect_day("Saturday को भेजना") == "saturday"
    assert detect_day("परसों") == "day_after"
    assert detect_day("कल कर दीजिए") == "tomorrow"


def test_wrong_pincode_is_corrected_by_readback():
    """The agent mishears once; the caller says no at read-back and the second try succeeds."""
    scenario = next(s for s in load_scenarios(SCENARIOS) if s["id"] == "confirm_hindi_digits")
    heard = iter(["हाँ", "पाँच छह शून्य शून्य शून्य दो", "नहीं, गलत है", "पाँच छह शून्य शून्य शून्य एक", "हाँ, सही है"])
    call = run_call(scenario, lambda line, i: (next(heard), 0.0, 0.0))
    assert call["passed"], call


def test_unintelligible_caller_hands_off_instead_of_acting():
    scenario = next(s for s in load_scenarios(SCENARIOS) if s["id"] == "cancel_hindi")
    call = run_call(scenario, lambda line, i: ("xyz", 0.0, 0.0))
    assert call["verdict"] == "handoff"
    assert call["outcome"] == "handoff"


def test_every_scenario_passes_with_perfect_hearing():
    for scenario in load_scenarios(SCENARIOS):
        call = run_call(scenario, oracle_hear)
        assert call["passed"], (scenario["id"], call["verdict"], call["outcome"])
