#!/bin/bash

export VOSK_SERVER_INTERFACE=127.0.0.1
export VOSK_MODEL_PATH=../models/vosk-model-ru-0.22
export VOSK_SPK_MODEL_PATH=../models/vosk-model-spk-0.4
export VOSK_SAMPLE_RATE=16000
export VOSK_SHOW_WORDS=true

python asr_server.py
