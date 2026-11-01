import re
from pathlib import Path

import numpy as np
import pytest

from parakh import audio
from parakh.run import all_lines, load_scenarios

RATE = audio.SAMPLE_RATE
SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios" / "cod_confirmation.yaml"


def tone(hz: float, seconds: float = 1.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def power(x: np.ndarray) -> float:
    return float(np.mean(x ** 2))


def test_phone_line_keeps_voice_band_and_removes_the_rest():
    assert power(audio.phone_line(tone(1000))) > 0.5 * power(tone(1000))
    assert power(audio.phone_line(tone(5000))) < 0.01 * power(tone(5000))
    assert power(audio.phone_line(tone(100))) < 0.1 * power(tone(100))


def test_phone_line_keeps_length():
    assert abs(len(audio.phone_line(tone(1000))) - RATE) <= 2


def test_add_noise_hits_requested_snr():
    speech = tone(1000)
    noise = np.random.default_rng(0).standard_normal(len(speech))
    mixed = audio._add_noise(speech, noise, 10)
    assert 10 * np.log10(power(speech) / power(mixed - speech)) == pytest.approx(10, abs=0.01)


@pytest.mark.parametrize("condition", list(audio.CONDITIONS))
def test_degrade_is_reproducible_and_valid(condition):
    speech, pool = tone(800), [tone(300), tone(500)]
    a = audio.degrade(speech, condition, seed=7, pool=pool)
    b = audio.degrade(speech, condition, seed=7, pool=pool)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()
    assert np.abs(a).max() <= 1.01


def test_noise_changes_with_seed():
    speech, pool = tone(800), [tone(300), tone(500)]
    a = audio.degrade(speech, "phone_street_10db", seed=1, pool=pool)
    b = audio.degrade(speech, "phone_street_10db", seed=2, pool=pool)
    assert not np.array_equal(a, b)


def test_every_voice_and_line_is_collected():
    lines = all_lines(load_scenarios(SCENARIOS))
    assert ("हाँ, भेज दीजिए", "hi-IN-MadhurNeural") in lines  # second entry of a list reply
    assert all(text and voice for text, voice in lines)


def test_english_voices_are_never_given_devanagari():
    """The speech service returns no audio for some Devanagari text on en-* voices."""
    for text, voice in all_lines(load_scenarios(SCENARIOS)):
        if voice.startswith("en-"):
            assert not re.search(r"[ऀ-ॿ]", text), (voice, text)
