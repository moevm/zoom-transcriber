import json
import subprocess
import time
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import ffmpeg
from vosk import KaldiRecognizer, Model, SetLogLevel

SetLogLevel(0)


class VoskResult(NamedTuple):
    start: float
    end: float
    text: str


@lru_cache(maxsize=None)
def get_model(model_path: str) -> KaldiRecognizer:
    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)
    return rec


def process_vosk_result(res: dict) -> VoskResult:
    if not (words := res.get("result")):
        return

    return VoskResult(start=words[0]["start"], end=words[-1]["end"], text=res["text"])


def process_audiofile(rec: KaldiRecognizer, file_path: str) -> None:
    result: list[VoskResult] = []

    _path = Path(file_path).absolute()
    print(f"Started processing {file_path}")
    start = time.perf_counter()
    process: subprocess.Popen = (
        ffmpeg.input(_path)
        .filter("highpass", f=200)
        .filter("lowpass", f=3000)
        .output("-", format="s16le", ac=1, ar=16000)
        .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
    )

    while data := process.stdout.read(4000):
        if rec.AcceptWaveform(data):
            recognized_data = json.loads(rec.Result())
            result.append(process_vosk_result(recognized_data))

    recognized_data = json.loads(rec.FinalResult())
    result.append(process_vosk_result(recognized_data))

    with open(_path.stem + ".json", "w", encoding="utf-8") as f:
        json.dump([r._asdict() for r in result], f, ensure_ascii=False, indent=2)

    print(
        f"Finished processing {file_path}, time elapsed: {time.perf_counter() - start}s"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="path to audio file")
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="models/vosk-model-small-ru-0.22",
        help="path to model",
    )
    args = parser.parse_args()

    model = get_model(args.model)
    process_audiofile(model, args.file)
