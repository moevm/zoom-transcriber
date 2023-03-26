import datetime
import functools
import json
import logging
import os
import sys
import time
from argparse import ArgumentParser
from functools import lru_cache
from typing import Dict, List, Tuple, Union

import numpy
import torch
from pyannote.audio import Pipeline
from whisper import Whisper, load_model
from whisper.audio import SAMPLE_RATE, load_audio

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(levelname)s]:%(pathname)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__file__)


@lru_cache(maxsize=None)
def get_pyannote_pipeline(config_path: str) -> Pipeline:
    return Pipeline.from_pretrained(config_path)


def get_whisper_model(model_type: str) -> Whisper:
    return load_model(model_type)


def _is_overlapping(a, b, th=0.35):
    s1, e1 = a
    s2, e2 = b
    if (s2 >= s1 and e2 <= s1) or (s1 >= s2 and e1 <= e2):
        return True
    return s2 <= e1 and e2 >= s1


def measure_duration(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        result = f(*args, **kwargs)
        end = time.monotonic()
        return datetime.timedelta(seconds=end - start), result

    return wrapper


def prepare_audio_file(file_path: str):
    full_path = os.path.abspath(file_path)
    return full_path, torch.from_numpy(load_audio(file_path))


@measure_duration
def identify_speakers(pipeline: Pipeline, audio: torch.Tensor):
    if not torch.is_tensor(audio):
        raise ValueError(
            f"Provided audio is not of type `torch.Tensor`, prepare the audio using the `prepare_audio_file` function"
        )

    # pyannote.audio accepts Tensor of shape [<channels_num>, <time>], in our case we read audio as mono waveform always
    audio_in_memory = {
        "waveform": torch.reshape(audio, (1, -1)),
        "sample_rate": SAMPLE_RATE,
    }
    speaker_diarization = pipeline(audio_in_memory)

    speakers_data = []

    for speech_turn, _, speaker in speaker_diarization.itertracks(yield_label=True):
        speakers_data.append(
            (
                float(f"{speech_turn.start:.1f}"),
                float(f"{speech_turn.end:.1f}"),
                speaker,
            )
        )
    return speakers_data


@measure_duration
def transcribe_recording(
    model: Whisper,
    audio: Union[str, torch.Tensor, numpy.ndarray],
    lang: str = "russian",
    no_speech_th=0.8,
) -> Dict:
    result = model.transcribe(
        audio=audio,
        verbose=False,
        word_timestamps=True,
        no_speech_threshold=no_speech_th,
        condition_on_previous_text=False,
        language="russian",
    )

    segments = result["segments"]
    lang = result["language"]

    segments = sorted(
        [s for s in segments if s["no_speech_prob"] <= no_speech_th],
        key=lambda s: s["start"],
    )
    text = " ".join(s["text"] for s in segments)
    return {"text": text, "segments": segments, "language": lang}


@measure_duration
def annotate_transcribed_result(
    transcribed_result: Dict, speaker_intervals: List[Tuple[float, float, str]]
) -> Dict:
    transcribed_result_annotated = []

    for transribed_segment in transcribed_result["segments"]:
        for speaker_segment in speaker_intervals:
            speaker_interval, speaker_label = speaker_segment[:2], speaker_segment[2]
            if _is_overlapping(
                [transribed_segment["start"], transribed_segment["end"]],
                speaker_interval,
            ):
                speaker = speaker_label
                break
        else:
            speaker = "UNKNOWN"

        transcribed_result_annotated.append(
            {
                "text": transribed_segment["text"],
                "start": transribed_segment["start"],
                "end": transribed_segment["end"],
                "speaker": speaker,
            }
        )

    return transcribed_result_annotated


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "-f", "--file", type=str, help="path to audio recording to process"
    )

    args = vars(parser.parse_args())

    logging.info("CUDA availability: %s", torch.cuda.is_available())

    logger.info("Loading whisper model")
    model = get_whisper_model("small")

    logger.info("Loading pyannote.audio pipeline from './models/config.yaml'")
    pyannote_pipline = get_pyannote_pipeline("./models/config.yaml")

    logger.info("Preparing input recording %s", args["file"])
    full_path, audio = prepare_audio_file(args["file"])

    logger.info("Transcribing input audio...")
    tr_duration, transribed_data = transcribe_recording(model, audio)

    logger.info("Identifying speakers...")
    sp_duration, speaker_invervals = identify_speakers(pyannote_pipline, audio)

    logger.info("Preparing the final result...")
    an_duration, annotated_result = annotate_transcribed_result(
        transribed_data, speaker_invervals
    )

    dir_name = os.path.dirname(full_path)
    file_name, _ = os.path.splitext(os.path.basename(full_path))
    result_json = os.path.join(dir_name, f"{file_name}.json")

    with open(result_json, "w") as f:
        json.dump(annotated_result, f, indent=2, ensure_ascii=False)

    logger.info("All done! Saved result to %s", result_json)

    print("Time taken:")
    print(f"Transcribing - {str(tr_duration)}")
    print(f"Speaker diarization - {str(sp_duration)}")
    print(f"Annotating - {str(an_duration)}")
