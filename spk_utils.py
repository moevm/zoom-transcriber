from typing import NamedTuple, Optional

import numpy as np


def cosine_dist(x, y):
    nx = np.array(x)
    ny = np.array(y)
    return 1 - np.dot(nx, ny) / (np.linalg.norm(nx) * np.linalg.norm(ny))


class SpkResult(NamedTuple):
    spk: Optional[list]
    spk_frames: Optional[float]


def extract_spk_from_result(res: dict, *args) -> SpkResult:
    if "spk" not in res:
        return

    return SpkResult(
        spk=res["spk"],
        spk_frames=res["spk_frames"],
    )


def check_spk_distance(res: dict, spk_vectors: list[list[float]]) -> bool:
    dists = [cosine_dist(res["spk"], vec) for vec in spk_vectors]
    avg_dist = np.mean(dists)
    # dists = [d <= 0.35 for d in dists]
    return avg_dist
