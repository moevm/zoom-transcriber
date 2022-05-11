from typing import NamedTuple, Optional

import numpy as np

from vosk_utils.utils import get_bad_words

__all__ = ["VoskResult", "SpkResult", "process_vosk_result", "extract_spk_from_result"]


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
        if len(dists) == len(names):
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
