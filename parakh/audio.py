"""Synthesise caller speech and degrade it the way a real phone call does.

Conditions get progressively harsher. All noise is seeded, so a run can be
reproduced exactly. These are simulations of a phone line and of background
noise, not recordings of real calls.
"""

import asyncio
import hashlib
from pathlib import Path

import edge_tts
import numpy as np
from faster_whisper import decode_audio
from scipy.signal import butter, resample_poly, sosfilt

SAMPLE_RATE = 16000
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "tts"

# name -> (use phone line?, noise type, signal-to-noise ratio in dB), mildest first
CONDITIONS = {
    "clean": (False, None, None),
    "phone": (True, None, None),
    "phone_street_10db": (True, "street", 10),
    "phone_crowd_5db": (True, "crowd", 5),
}


# A caller who repeats a line does not produce the same audio twice, so each try is spoken at a
# different speed. Without this, a retry on clean audio is the identical clip and adds nothing.
RETRY_RATES = ("+0%", "-10%", "+10%")


def _cache_path(text: str, voice: str, rate: str = "+0%") -> Path:
    key = hashlib.sha256(f"{voice}|{text}".encode() if rate == "+0%" else f"{voice}|{text}|{rate}".encode())
    return CACHE_DIR / f"{key.hexdigest()[:20]}.mp3"


async def _synthesise_all(items: list[tuple[str, str, str]], concurrency: int = 4) -> None:
    gate = asyncio.Semaphore(concurrency)

    async def one(text, voice, rate):
        path = _cache_path(text, voice, rate)
        if path.exists():
            return
        # A failed save leaves an empty file behind, so only a finished clip gets the real name.
        partial = path.with_suffix(".part")
        async with gate:
            for attempt in range(4):
                try:
                    await edge_tts.Communicate(text, voice, rate=rate).save(str(partial))
                    partial.replace(path)
                    return
                except Exception as error:
                    if attempt == 3:
                        raise RuntimeError(f"Could not synthesise {text!r} with voice {voice}") from error
                    await asyncio.sleep(2 ** attempt)

    await asyncio.gather(*(one(*item) for item in items))


def synthesise_all(items: list[tuple[str, str]], rates=RETRY_RATES) -> None:
    """Fetch every (text, voice) pair at each speed once; later runs reuse the cached files."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(_synthesise_all([(t, v, r) for t, v in sorted(set(items)) for r in rates]))


def load_speech(text: str, voice: str, rate: str = "+0%") -> np.ndarray:
    return decode_audio(str(_cache_path(text, voice, rate)), sampling_rate=SAMPLE_RATE).astype(np.float32)


def seed_for(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def phone_line(audio: np.ndarray) -> np.ndarray:
    """Narrowband telephone channel: 8 kHz sampling, 300-3400 Hz band, 8-bit mu-law."""
    narrow = resample_poly(audio, 1, 2)
    band = butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos")
    narrow = sosfilt(band, narrow)
    mu = 255.0
    narrow = np.clip(narrow, -1, 1)
    coded = np.round(np.sign(narrow) * np.log1p(mu * np.abs(narrow)) / np.log1p(mu) * 127) / 127
    decoded = np.sign(coded) * ((1 + mu) ** np.abs(coded) - 1) / mu
    return resample_poly(decoded, 2, 1).astype(np.float32)


def _street_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink noise with extra low-frequency rumble, standing in for traffic."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    freqs[0] = freqs[1]
    spectrum *= 1 / np.sqrt(freqs) * (1 + 3 * (freqs < 200))
    return np.fft.irfft(spectrum, n)


def _crowd_noise(n: int, rng: np.random.Generator, pool: list[np.ndarray]) -> np.ndarray:
    """Several other callers talking at once: real speech is the hardest noise to separate."""
    mix = np.zeros(n)
    for _ in range(6):
        clip = pool[rng.integers(len(pool))]
        repeated = np.tile(clip, n // len(clip) + 2)
        start = rng.integers(len(clip))
        mix += repeated[start:start + n]
    return mix


def _add_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    if speech_power == 0 or noise_power == 0:
        return speech
    scaled = noise * np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    return speech + scaled


def degrade(speech: np.ndarray, condition: str, seed: int, pool: list[np.ndarray]) -> np.ndarray:
    use_phone, noise_type, snr_db = CONDITIONS[condition]
    out = speech
    if noise_type:
        rng = np.random.default_rng(seed)
        noise = _street_noise(len(out), rng) if noise_type == "street" else _crowd_noise(len(out), rng, pool)
        out = _add_noise(out, noise, snr_db)
    peak = np.max(np.abs(out))
    if peak > 1:
        out = out / peak
    return phone_line(out) if use_phone else out.astype(np.float32)
