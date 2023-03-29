import pickle
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

BAD_WORDS_PATH = str(Path(__file__).parent.absolute().joinpath("bad_words.pkl"))


# TODO: make async verision (async for)
def merge_iterable(
    iterable: Iterable[Any],
    is_close: Callable[[Any, Any], bool],
    merge: Callable[[Any, Any], Any],
    apply: Optional[Callable[[Any], Any]] = None,
) -> Iterable[Any]:
    """Merges elements in sorted iterable"""
    merged = []
    for el in iterable:
        if apply is not None:
            el = apply(el)
        # if the list of merged elements is empty or if the current
        # element is not close with the previous, simply append it.
        if not merged or not is_close(merged[-1], el):
            merged.append(el)
        else:
            # otherwise, elements can be merged
            merged[-1] = merge(merged[-1], el)

    return merged


def get_bad_words():
    with open(BAD_WORDS_PATH, "rb") as f:
        bad_words = pickle.load(f)

    if not isinstance(bad_words, Iterable):
        raise ValueError(f"Bad words should be an iterble, got {type(bad_words)}")

    return bad_words
