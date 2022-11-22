import logging
import wave
from pathlib import Path
from typing import NamedTuple, Optional

import noisereduce as nr
import numpy as np
import python_speech_features as psf
from numpy.typing import NDArray
from pysndfx import AudioEffectsChain
from scipy import signal

from vosk_utils.utils import get_bad_words

__all__ = ["VoskResult", "SpkResult", "process_vosk_result", "extract_spk_from_result"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BAD_WORDS = get_bad_words()


QUESTION_RULES = [
    lambda text: "вопрос" in text,
    lambda text: "а что такое" in text,
    lambda text: "в чем заключается" in text,
]


class VoskResult(NamedTuple):
    start: float
    end: float
    conf: float
    text: str
    speaker: str
    is_question: bool


class SpkResult(NamedTuple):
    text: str
    spk: Optional[list]
    spk_frames: Optional[float]


def process_vosk_result(
    res: dict, spk_vectors: dict[str, list[list[float]]]
) -> VoskResult:
    if not (words := res.get("result")):
        return

    text: str = res["text"]

    is_question = False
    for rule in QUESTION_RULES:
        if rule(text):
            is_question = True
            break

    for bad_word in BAD_WORDS:
        text = text.replace(bad_word, "")

    speaker_name = "unknown"
    if "spk" in res:
        names = tuple(spk_vectors.keys())
        dists = list(
            filter(
                None, (check_spk_distance(res, vecs) for vecs in spk_vectors.values())
            )
        )
        l_d = len(dists)
        if l_d == len(names) and l_d > 0:
            speaker_name = names[np.argmin(dists)]

    conf = np.mean([w["conf"] for w in words])
    return VoskResult(
        start=words[0]["start"],
        end=words[-1]["end"],
        conf=conf,
        text=text,
        speaker=speaker_name,
        is_question=is_question,
    )


def cosine_dist(x, y):
    nx = np.array(x)
    ny = np.array(y)
    return 1 - np.dot(nx, ny) / (np.linalg.norm(nx) * np.linalg.norm(ny))


def extract_spk_from_result(res: dict, *args) -> SpkResult:
    if "spk" not in res:
        return

    return SpkResult(
        text=res["text"],
        spk=res["spk"],
        spk_frames=res["spk_frames"],
    )


def check_spk_distance(res: dict, spk_vectors: list[list[float]]) -> bool:
    dists = [cosine_dist(res["spk"], vec) for vec in spk_vectors]
    avg_dist = np.mean(dists)
    # dists = [d <= 0.35 for d in dists]
    return avg_dist


class Denoiser:
    def __init__(self, samplerate: int, channels: int):
        self.samplerate = samplerate
        self.channels = channels
        logger.info("Denoising with samplerate=%s, channels=%s", samplerate, channels)

    def reduce_noise(self, audio_clip: NDArray):
        return nr.reduce_noise(
            y=audio_clip, time_mask_smooth_ms=64, n_fft=4096, sr=self.samplerate
        )

    def _bytes_audio_to_numpy(self, audio: bytes) -> NDArray:
        array = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        return array

    def _numpy_to_bytes(self, data: NDArray) -> bytes:
        return data.astype(np.int16).tobytes()

    def reduce_noise_power(self, y):
        return signal.medfilt(y, 3)

    def denoise_audio_chunk_raw(self, audio: bytes) -> bytes:
        audio_chunk = self._bytes_audio_to_numpy(audio)
        audio_chunk = self.reduce_noise(audio_chunk)
        return self._numpy_to_bytes(audio_chunk)

    def __call__(self, audio: bytes) -> bytes:
        return self.denoise_audio_chunk_raw(audio)
