from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from backend.config import DB_NAME, MONGODB_URI, SESSIONS_COL_NAME, SPEAKERS_COL_NAME


def get_db(db_name) -> AsyncIOMotorDatabase:
    return AsyncIOMotorClient(MONGODB_URI)[db_name]


db = get_db(DB_NAME)
speakers_col = db[SPEAKERS_COL_NAME]
sessions_col = db[SESSIONS_COL_NAME]
