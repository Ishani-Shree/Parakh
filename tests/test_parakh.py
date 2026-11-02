from pathlib import Path

import pytest

from parakh.normalize import detect_day, detect_intent, detect_yes_no, extract_digits, similarity
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


@pytest.mark.parametrize("heard, intent", [
    ("जी सही हैं बेज दीजिये", "confirm"),
    ("हाँ भेज दीजिये!", "confirm"),
    ("नहीं नहीं, कानसल मत करो, बेज दो", "confirm"),
    ("नॉ, कानसल दिस अड़र।", "cancel"),
    ("अभी नहीं, मैं घर पर नहीं हूं.", "reschedule"),
    ("दिलिवरी दूसरे अड्रेस पर करनी हैं।", "change_address"),
])
def test_intent_survives_spelling_variants(heard, intent):
    assert detect_intent(heard) == intent


@pytest.mark.parametrize("risky", ["क्या है?", "है", "नहीं है", "कल"])
def test_folding_does_not_turn_other_words_into_yes(risky):
    assert detect_yes_no(risky) != "yes"


@pytest.mark.parametrize("text, digits", [
    ("साठ", "60"),      # must stay distinct from सात (7) after folding
    ("सात", "7"),
    ("आठ", "8"),
    ("तीस", "30"),
    ("तीन", "3"),
    ("दस", "10"),
    ("दो", "2"),
    ("छः", "6"),
    ("छह", "6"),
    ("ज़ीरो", "0"),
    ("जीरो", "0"),
])
def test_folding_keeps_numbers_distinct(text, digits):
    assert extract_digits(text) == digits


def test_similarity_separates_close_and_far_transcripts():
    assert similarity("हाँ, भेज दीजिए", "हाँ बेज दीजिये") > 0.7
    assert similarity("Okay, theek hai", "अकि तिखे") < 0.7


def test_every_held_out_scenario_passes_with_perfect_hearing():
    for scenario in load_scenarios(SCENARIOS.with_name("cod_holdout.yaml")):
        call = run_call(scenario, oracle_hear)
        assert call["passed"], (scenario["id"], call["verdict"], call["outcome"])


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
    call = run_call(scenario, lambda line, i, attempt: (next(heard), 0.0, 0.0))
    assert call["passed"], call


def test_unintelligible_caller_hands_off_instead_of_acting():
    scenario = next(s for s in load_scenarios(SCENARIOS) if s["id"] == "cancel_hindi")
    call = run_call(scenario, lambda line, i, attempt: ("xyz", 0.0, 0.0))
    assert call["verdict"] == "handoff"
    assert call["outcome"] == "handoff"


def test_every_scenario_passes_with_perfect_hearing():
    for scenario in load_scenarios(SCENARIOS):
        call = run_call(scenario, oracle_hear)
        assert call["passed"], (scenario["id"], call["verdict"], call["outcome"])
