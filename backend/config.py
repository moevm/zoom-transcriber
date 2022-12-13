import os
from pathlib import Path

# url where vosk websocket server is started
VOSK_SERVER_WS_URL = os.getenv("VOSK_SERVER_WS_URL", "ws://127.0.0.1:2700")
# url to mongodb
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
# name of mongo database
DB_NAME = os.getenv("MONGO_VOSK_DB_NAME", "nir-zoom")
# name for collection with speaker records
SPEAKERS_COL_NAME = os.getenv("MONGO_SPEAKERS_COL_NAME", "speakers")
# name for collection with meetings records
SESSIONS_COL_NAME = os.getenv("MONGO_SESSIONS_COL_NAME", "sessions")
# value for analyzing quality of recorded speaker features
GOOD_SPK_FRAMES_NUM = int(os.getenv("GOOD_SPK_FRAMES_NUM", "300"))
# value for checking quantity of recorded speaker features
MIN_SPK_VECTORS_NUM = int(os.getenv("MIN_SPK_VECTORS_NUM", "8"))
# value restricting min border of ratio = (num good speaker features / num all speaker features)
SPK_GOOD_RATIO = float(os.getenv("SPK_GOOD_RATIO", "0.65"))
# value in secs - how close should be two phrases to be combined into one
MERGE_DIFF_SEC = float(os.getenv("MERGE_DIFF_SEC", "2.5"))
ROOT_DIR = Path(__file__).absolute().parent
