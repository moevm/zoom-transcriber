import os
from pathlib import Path

VOSK_SERVER_WS_URL = os.getenv("VOSK_SERVER_WS_URL", "ws://localhost:2700")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_VOSK_DB_NAME", "nir-zoom")
SPEAKERS_COL_NAME = os.getenv("MONGO_SPEAKERS_COL_NAME", "speakers")
SESSIONS_COL_NAME = os.getenv("MONGO_SESSIONS_COL_NAME", "sessions")

GOOD_SPK_FRAMES_NUM = int(os.getenv("GOOD_SPK_FRAMES_NUM", "300"))
MIN_SPK_VECTORS_NUM = int(os.getenv("MIN_SPK_VECTORS_NUM", "8"))
SPK_GOOD_RATIO = float(os.getenv("SPK_GOOD_RATIO", "0.65"))
ROOT_DIR = Path(__file__).absolute().parent
MERGE_DIFF_SEC = float(os.getenv("MERGE_DIFF_SEC", "1.5"))
