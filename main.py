import json
import os
import subprocess
import textwrap
import time
from argparse import ArgumentParser, RawTextHelpFormatter
from concurrent.futures import Future, ThreadPoolExecutor, wait
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Optional, Union

from vosk import KaldiRecognizer, Model, SetLogLevel, SpkModel

from question_detection.ru import is_phrase_a_question
from utils import merge_iterable
from vosk_utils import (
    Denoiser,
    SpkResult,
    VoskResult,
    extract_spk_from_result,
    process_vosk_result,
)

SetLogLevel(-1)

NUM_ALTERNATIVES = 4
MERGE_DIFF_SEC = 2


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
    def __init__(self, loader, path: Optional[str] = None):
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
    reduce_noise: bool = False,
) -> tuple[list[VoskResult], str]:
    result: list[VoskResult] = []
    file_path = Path(file_path_raw).absolute()
    file_name = file_path.stem

    print(f"Started processing file {file_name}")
    start_time = time.time()

    sample_rate = 16000

    process = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "quiet",
            "-i",
            file_path,
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-f",
            "s16le",
            "-",
        ],
        bufsize=0,
        stdout=subprocess.PIPE,
    )

    rec = KaldiRecognizer(model, sample_rate)
    if spk_model:
        rec.SetSpkModel(spk_model)

    rec.SetWords(True)
    rec.Reset()

    i = 0
    read_n = 4000

    if reduce_noise:
        denoiser = Denoiser(samplerate=sample_rate, channels=1)
    else:
        denoiser = lambda x: x

    while bytes_data := process.stdout.read(read_n):
        denoised_bytes_data = denoiser(bytes_data)
        if rec.AcceptWaveform(denoised_bytes_data):
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


def apply(segment: VoskResult) -> VoskResult:
    return VoskResult(
        start=segment.start,
        end=segment.end,
        words=segment.words,
        conf=segment.conf,
        text=segment.text,
        speaker=segment.speaker,
        is_question=is_phrase_a_question(segment.text.strip()),
    )


def records_close(a: VoskResult, b: VoskResult):
    first_scenario = a.speaker == b.speaker and a.is_question == b.is_question
    second_scenario = (
        a.speaker == b.speaker and a.is_question and (b.start - a.end) <= MERGE_DIFF_SEC
    )
    return first_scenario or second_scenario


def records_merge(a: VoskResult, b: VoskResult):
    combined_words = (a.words or []) + (b.words or [])

    return VoskResult(
        start=a.start,
        end=b.end,
        words=combined_words,
        text=" ".join((a.text, b.text)).strip(),
        conf=(a.conf + b.conf) / 2,
        speaker=a.speaker,
        is_question=a.is_question or b.is_question,
    )


def worker(audio_file: str, reduce_noise: bool):
    fn = partial(process_vosk_result, spk_vectors=spk_vectors_loader.get())

    result, _ = process_audiofile(
        file_path_raw=audio_file,
        model=model_loader.get(),
        process_fn=fn,
        spk_model=spk_model_loader.get(),
        reduce_noise=reduce_noise,
    )
    return result


def extract_spk_worker(audio_file: str, reduce_noise: bool):
    speaker_result, speaker_name = process_audiofile(
        file_path_raw=audio_file,
        model=model_loader.get(),
        process_fn=extract_spk_from_result,
        spk_model=spk_model_loader.get(),
        reduce_noise=reduce_noise,
    )
    return speaker_result, speaker_name


def process_dir(
    dir_path_raw: str,
    model_path: str,
    spk_model_path: str = None,
    spk_vectors_path: str = None,
    reduce_noise: bool = False,
    patterns: list[str] = ["*.m4a", "*.mp3", "*.wav"],
) -> None:
    dir_path = Path(dir_path_raw).absolute()
    dir_name = dir_path.stem
    audio_file_paths = (f for pattern in patterns for f in dir_path.glob(pattern))
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

    futures: list[Future] = []
    with ThreadPoolExecutor(4) as pool:
        for path in audio_file_paths:
            futures.append(pool.submit(worker, path, reduce_noise))
        wait(futures)

    res = [f.result() for f in futures]

    meeting_result = [x for y in res for x in y]  # flatten

    meeting_result.sort(key=lambda r: r.start)

    meeting_result = merge_iterable(
        meeting_result, records_close, records_merge, apply=apply
    )

    with open(dir_name + "/result.json", "w", encoding="utf-8") as f:
        json.dump(
            [r._asdict() for r in meeting_result], f, ensure_ascii=False, indent=2
        )

    elapsed_time = time.time() - start_time
    print(f"Finished processing directory {dir_path}, elapsed time: {elapsed_time}s")


def process_extract_spk_dir(
    dir_path_raw: str,
    model_path: str,
    spk_model_path: str = None,
    reduce_noise: bool = False,
    patterns: list[str] = ["*.m4a", "*.mp3", "*.wav"],
) -> None:
    dir_path = Path(dir_path_raw).absolute()
    dir_name = dir_path.stem
    audio_file_paths = (f for pattern in patterns for f in dir_path.glob(pattern))
    os.makedirs(dir_name, exist_ok=True)
    print(f"Started processing directory {dir_name} in spk mode")
    start_time = time.time()

    model_loader.set_path(model_path)
    model_loader.load()

    if spk_model_path:
        spk_model_loader.set_path(spk_model_path)
        spk_model_loader.load()

    futures: list[Future] = []
    with ThreadPoolExecutor(4) as pool:
        for path in audio_file_paths:
            futures.append(pool.submit(extract_spk_worker, path, reduce_noise))

        wait(futures)

    res: list[tuple[list[SpkResult], str]] = [f.result() for f in futures]

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
    parser = ArgumentParser(
        description=textwrap.dedent(
            """
    CLI app for transcribing audio files for speaker and question detection.
    Can run in 2 modes:
    1) default mode - processes all mp4 files in provided dir and creates a single result.json file with all detected phrases in timestamp order from the start. To process multispeaker recording provided dir should contain single audio file with combined sound.
    2) spk extraction mode - prepares speaker features data for default mode, provided dir should contain speaker recording, with one file for each speaker about 2-5 minutes long (with long sentences and pauses about 2 secons between each sentence).
    """
        ),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to directory with m4a audio files, for speaker features extraction filenames would be used as speaker names. App would create the folder with the same name as provided dir in current folder to write results to: `spk_vectors.json` in case of spk extraction, `result.json` in case of default processing mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="models/vosk-model-small-ru-0.22",
        help="path to Vosk language model",
    )
    parser.add_argument(
        "--spk-model",
        type=str,
        required=False,
        default=None,
        help="path to Vosk spk model",
    )
    parser.add_argument(
        "--spk-vectors",
        type=str,
        required=False,
        default=None,
        help="Path to spk_vectors.json file, containing info about speaker names and feauteres. Can be prepared in --extract-spk mode, and required for default processing mode",
    )
    parser.add_argument(
        "--extract-spk",
        action="store_true",
        help="if set, runs in spk extraction mode",
    )
    parser.add_argument(
        "--reduce_noise",
        action="store_true",
        help="if set, perform noise reduction",
    )
    args = parser.parse_args()

    dir_path = args.dir
    if args.spk_model is not None:
        args.spk_model = os.path.abspath(args.spk_model)

    if args.extract_spk:
        process_extract_spk_dir(
            dir_path,
            os.path.abspath(args.model),
            args.spk_model,
            args.reduce_noise,
        )
    else:
        if not args.spk_vectors:
            print("No spk vectors path provided")
            exit(1)

        args.spk_vectors = os.path.abspath(args.spk_vectors)

        process_dir(
            dir_path,
            os.path.abspath(args.model),
            args.spk_model,
            args.spk_vectors,
            args.reduce_noise,
        )
