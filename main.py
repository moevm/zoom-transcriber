import json
import subprocess
from argparse import ArgumentParser
from typing import NamedTuple

import ffmpeg
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(0)


class VoskResult(NamedTuple):
    start: float
    end: float
    text: str


def process_vosk_result(res: dict) -> VoskResult:
    if not (words := res.get("result")):
        return

    return VoskResult(
        start=words[0]["start"],
        end=words[-1]["end"],
        text=res["text"]
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="path to audio file")
    parser.add_argument("--model", type=str, required=False, default="models/vosk-model-small-ru-0.22", help="path to model")
    args = parser.parse_args()
    
    samplerate = 16000

    model = Model(args.model)
    rec = KaldiRecognizer(model, samplerate)
    rec.SetWords(True)
    
    process: subprocess.Popen = (
        ffmpeg
        .input(args.file)
        .filter("highpass", f=200)
        .filter("lowpass", f=3000)
        .output('-', format='s16le', ac=1, ar=samplerate)
        .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
    )

    while data := process.stdout.read(4000):
        if rec.AcceptWaveform(data):
            recognized_data = json.loads(rec.Result())
            print(process_vosk_result(recognized_data))

    recognized_data = json.loads(rec.FinalResult())
    print(process_vosk_result(recognized_data))
    