"""Local speech-to-text with faster-whisper.

A config is a model name with an optional "+hint", which primes Whisper with the
vocabulary this kind of call uses (for example "small+hint").
"""

import os
import time

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless on Windows, just noisy

# A plain sentence about the call's topic. Listing expected answers would make Whisper echo them
# on unclear audio, which passes tests for the wrong reason.
DOMAIN_HINT = "फ़ोन पर ग्राहक से ऑनलाइन ऑर्डर की डिलीवरी के बारे में बातचीत।"


class Transcriber:
    def __init__(self, config: str, device: str = "cuda"):
        # CTranslate2 needs cuBLAS on PATH; point this at a folder containing it if needed.
        if dll_dir := os.environ.get("PARAKH_CUDA_DLL_DIR"):
            os.add_dll_directory(dll_dir)
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]
        from faster_whisper import WhisperModel

        self.config = config
        name, _, flag = config.partition("+")
        self.hint = DOMAIN_HINT if flag == "hint" else None
        compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(name, device=device, compute_type=compute)
        self.transcribe(np.zeros(16000, dtype=np.float32))  # the first call is slow; keep it out of the results

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Return (text, seconds). beam_size=1 because a live call cannot wait for a wider search."""
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(
            audio, language="hi", beam_size=1, condition_on_previous_text=False,
            initial_prompt=self.hint, vad_filter=False,
        )
        text = " ".join(s.text.strip() for s in segments)  # segments is lazy: decoding happens here
        return text, time.perf_counter() - t0
