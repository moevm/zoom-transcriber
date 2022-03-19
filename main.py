import json
import os
import subprocess
import time
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Union

from vosk import KaldiRecognizer, Model, SetLogLevel

SetLogLevel(-1)


QUESTION_RULES = [
    lambda text: "вопрос" in text and "спасибо" not in text,
    lambda text: "а что такое" in text,
    lambda text: "в чем заключается" in text,
]


class VoskResult(NamedTuple):
    start: float
    end: float
    text: str
    speaker: str
    is_question: bool


@lru_cache(maxsize=5)
def get_model(model_path: str) -> KaldiRecognizer:
    print(f"Init {model_path} model loading")
    model = Model(model_path)
    print("Model loading completed")
    return model


def process_vosk_result(res: dict, speaker_name: str) -> Model:
    if not (words := res.get("result")):
        return

    text = res["text"]
    is_question = False
    for rule in QUESTION_RULES:
        if rule(text):
            is_question = True
            break

    return VoskResult(
        start=words[0]["start"],
        end=words[-1]["end"],
        text=text,
        speaker=speaker_name,
        is_question=is_question,
    )


def process_audiofile(
    file_path_raw: Union[str, Path], model: Model, speaker_name: str = None
) -> tuple[list[VoskResult], str]:
    result: list[VoskResult] = []
    file_path = Path(file_path_raw).absolute()
    file_name = file_path.stem

    print(f"Started processing file {file_name}")
    start_time = time.time()

    process = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "quiet",
            "-i",
            file_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-filter:a",
            "highpass=f=200, lowpass=f=3000",
            "-f",
            "s16le",
            "-",
        ],
        bufsize=0,
        stdout=subprocess.PIPE,
    )

    if speaker_name is None:
        speaker_name = file_name

    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)
    rec.Reset()

    i = 0
    while bytes_data := process.stdout.read(4000):
        if rec.AcceptWaveform(bytes_data):
            recognized_data = json.loads(rec.Result())
            processed_result = process_vosk_result(recognized_data, speaker_name)
            if processed_result is not None:
                result.append(processed_result)

        if i % 100 == 0:
            print(f"processed data chunk #{i}")

        i += 1

    process.wait()

    recognized_data = json.loads(rec.FinalResult())
    processed_result = process_vosk_result(recognized_data, speaker_name)
    if processed_result is not None:
        result.append(processed_result)

    elapsed_time = time.time() - start_time
    print(f"Finished processig file {file_name}, elapsed time {elapsed_time}s")

    return result, file_name


def process_dir(dir_path_raw: str, model_path: str) -> None:
    dir_path = Path(dir_path_raw).absolute()
    dir_name = dir_path.stem
    audio_file_paths = dir_path.glob("*.m4a")
    os.makedirs(dir_name, exist_ok=True)
    print(f"Started processing directory {dir_name}")
    start_time = time.time()
    meeting_result: list[VoskResult] = []
    for audio_file in audio_file_paths:
        speaker_result, _ = process_audiofile(audio_file, get_model(model_path))
        meeting_result.extend(speaker_result)

    meeting_result.sort(key=lambda r: r.start)

    with open(dir_name + "/result.json", "w", encoding="utf-8") as f:
        json.dump(
            [r._asdict() for r in meeting_result], f, ensure_ascii=False, indent=2
        )

    elapsed_time = time.time() - start_time
    print(f"Finished processing directory {dir_path}, elapsed time: {elapsed_time}s")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dir", type=str, required=True, help="path to directory with m4a audio files"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="models/vosk-model-small-ru-0.22",
        help="path to model",
    )
    args = parser.parse_args()

    dir_path = args.dir

    process_dir(dir_path, os.path.abspath(args.model))
