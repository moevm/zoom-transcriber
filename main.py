import json
import os
import subprocess
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Optional, Union

from vosk import KaldiRecognizer, Model, SetLogLevel, SpkModel

from vosk_utils import (
    SpkResult,
    VoskResult,
    extract_spk_from_result,
    process_vosk_result,
)
from vosk_utils.utils import merge_iterable

SetLogLevel(0)

NUM_ALTERNATIVES = 4
MERGE_DIFF_SEC = 0.75


@lru_cache(maxsize=5)
def _get_model(model_path: str) -> Model:
    print(f"Init {model_path} model loading")
    model = Model(model_path)
    print("Model loading completed")
    return model


@lru_cache(maxsize=5)
def _get_spk_model(spk_model_path: Optional[str] = None) -> SpkModel:
    if spk_model_path is not None:
        print(f"Init spk {spk_model_path} model loading")
        model = SpkModel(spk_model_path)
        print(f"Spk model loading completed")
        return model


@lru_cache(maxsize=5)
def _get_spk_vectors(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class Loader:
    def __init__(self, loader, path: str = None):
        self.path = path
        self.loader = loader
        self._data = None

    def set_path(self, path: str) -> None:
        self.path = path

    def load(self):
        if not self.path:
            raise AttributeError("Model path is not set")

        self._data = self.loader(self.path)

    def get(self):
        return self._data


model_loader = Loader(_get_model)
spk_model_loader = Loader(_get_spk_model)
spk_vectors_loader = Loader(_get_spk_vectors)


def process_audiofile(
    file_path_raw: Union[str, Path],
    model: Model,
    process_fn: Callable[[dict, str], Union[VoskResult, SpkResult]],
    spk_model: Optional[SpkModel] = None,
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
            "-f",
            "s16le",
            "-",
        ],
        bufsize=0,
        stdout=subprocess.PIPE,
    )

    # speaker_name = file_name

    rec = KaldiRecognizer(model, 16000)
    if spk_model:
        rec.SetSpkModel(spk_model)

    rec.SetWords(True)
    rec.Reset()

    i = 0
    while bytes_data := process.stdout.read(4000):
        if rec.AcceptWaveform(bytes_data):
            recognized_data = json.loads(rec.Result())
            processed_result = process_fn(recognized_data)
            if processed_result is not None:
                result.append(processed_result)

        if i % 250 == 0:
            print(f"{file_name=}, processed data chunk #{i}")

        i += 1

    process.wait()

    recognized_data = json.loads(rec.FinalResult())
    processed_result = process_fn(recognized_data)
    if processed_result is not None:
        result.append(processed_result)

    elapsed_time = time.time() - start_time
    print(f"Finished processig file {file_name}, elapsed time {elapsed_time}s")

    return result, file_name


def records_close(a: VoskResult, b: VoskResult):
    return a.speaker == b.speaker and abs(b.start - a.end) < MERGE_DIFF_SEC


def records_merge(a: VoskResult, b: VoskResult):
    return VoskResult(
        start=a.start,
        end=b.end,
        text=" ".join((a.text, b.text)).strip(),
        speaker=a.speaker,
        is_question=a.is_question and b.is_question,
    )


def worker(audio_file):
    fn = partial(process_vosk_result, spk_vectors=spk_vectors_loader.get())

    result, _ = process_audiofile(
        file_path_raw=audio_file,
        model=model_loader.get(),
        process_fn=fn,
        spk_model=spk_model_loader.get(),
    )
    return result


def extract_spk_worker(audio_file):
    speaker_result, speaker_name = process_audiofile(
        file_path_raw=audio_file,
        model=model_loader.get(),
        process_fn=extract_spk_from_result,
        spk_model=spk_model_loader.get(),
    )
    return speaker_result, speaker_name


def process_dir(
    dir_path_raw: str,
    model_path: str,
    spk_model_path: str = None,
    spk_vectors_path: str = None,
) -> None:
    dir_path = Path(dir_path_raw).absolute()
    dir_name = dir_path.stem
    audio_file_paths = dir_path.glob("*.m4a")
    os.makedirs(dir_name, exist_ok=True)
    print(f"Started processing directory {dir_name}")
    start_time = time.time()
    meeting_result: list[VoskResult] = []

    model_loader.set_path(model_path)
    model_loader.load()

    if spk_model_path:
        spk_model_loader.set_path(spk_model_path)
        spk_model_loader.load()

    if spk_vectors_path:
        spk_vectors_loader.set_path(spk_vectors_path)
        spk_vectors_loader.load()

    with ThreadPoolExecutor(4) as pool:
        res = pool.map(worker, audio_file_paths)

    meeting_result = [x for y in res for x in y]  # flatten

    meeting_result.sort(key=lambda r: r.start)

    meeting_result = merge_iterable(meeting_result, records_close, records_merge)

    with open(dir_name + "/result.json", "w", encoding="utf-8") as f:
        json.dump(
            [r._asdict() for r in meeting_result], f, ensure_ascii=False, indent=2
        )

    elapsed_time = time.time() - start_time
    print(f"Finished processing directory {dir_path}, elapsed time: {elapsed_time}s")


def process_extract_spk_dir(
    dir_path_raw: str, model_path: str, spk_model_path: str = None
) -> None:
    dir_path = Path(dir_path_raw).absolute()
    dir_name = dir_path.stem
    audio_file_paths = dir_path.glob("*.m4a")
    os.makedirs(dir_name, exist_ok=True)
    print(f"Started processing directory {dir_name} in spk mode")
    start_time = time.time()

    model_loader.set_path(model_path)
    model_loader.load()

    if spk_model_path:
        spk_model_loader.set_path(spk_model_path)
        spk_model_loader.load()

    with ThreadPoolExecutor(4) as pool:
        res: list[tuple[list[SpkResult], str]] = pool.map(
            extract_spk_worker, audio_file_paths
        )

    spk_data = {}
    for spk_results, speaker_name in res:
        data = list(filter(lambda x: x is not None and x.spk_frames > 400, spk_results))
        data = [
            el.spk
            for el in sorted(data, key=lambda el: el.spk_frames, reverse=True)[:25]
        ]

        spk_data[speaker_name] = data

    with open(dir_name + "/spk_vectors.json", "w", encoding="utf-8") as f:
        json.dump(spk_data, f, ensure_ascii=False, indent=2)

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
    parser.add_argument(
        "--spk-model",
        type=str,
        required=False,
        default=None,
        help="path to spk model",
    )
    parser.add_argument(
        "--spk-vectors",
        type=str,
        required=False,
        default=None,
        help="path to spk vectors",
    )
    parser.add_argument(
        "--extract-spk",
        action="store_true",
        help="if set, runs in spk extraction mode",
    )
    args = parser.parse_args()

    dir_path = args.dir
    if args.spk_model is not None:
        args.spk_model = os.path.abspath(args.spk_model)

    if args.extract_spk:
        process_extract_spk_dir(dir_path, os.path.abspath(args.model), args.spk_model)
    else:
        if not args.spk_vectors:
            print("No spk vectors path provided")
            exit(1)

        args.spk_vectors = os.path.abspath(args.spk_vectors)

        process_dir(
            dir_path, os.path.abspath(args.model), args.spk_model, args.spk_vectors
        )
