import asyncio
import base64
import csv
import json
import math
import zipfile
from datetime import datetime
from functools import partial
from io import BytesIO, StringIO
from pathlib import Path
from typing import List, Optional, Union

import websockets
from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    staticfiles,
    status,
    templating,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.config import (
    GOOD_SPK_FRAMES_NUM,
    MERGE_DIFF_SEC,
    MIN_SPK_VECTORS_NUM,
    ROOT_DIR,
    SPK_GOOD_RATIO,
    VOSK_SERVER_WS_URL,
)
from backend.db import sessions_col, speakers_col
from vosk_utils import extract_spk_from_result, process_vosk_result

app = FastAPI(title="audio-transcriber")
templates = templating.Jinja2Templates(ROOT_DIR.joinpath("templates"))


async def get_spk_cookie_data_ws(
    websocket: WebSocket, spk_session: Optional[str] = Cookie(None)
) -> dict:
    if spk_session is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    return json.loads(base64.urlsafe_b64decode(spk_session).decode("utf-8"))


async def get_spk_cookie_data_http(spk_session: Optional[str] = Cookie(None)) -> dict:
    if spk_session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad session"
        )
    return json.loads(base64.urlsafe_b64decode(spk_session).decode("utf-8"))


async def get_meeting_cookie_data_ws(
    websocket: WebSocket, meeting_session: Optional[str] = Cookie(None)
) -> dict:
    if meeting_session is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    return json.loads(base64.urlsafe_b64decode(meeting_session).decode("utf-8"))


async def get_meeting_cookie_data_http(
    meeting_session: Optional[str] = Cookie(None),
) -> dict:
    if meeting_session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad session"
        )
    return json.loads(base64.urlsafe_b64decode(meeting_session).decode("utf-8"))


def create_session_cookie(session_id: str, start_date: datetime) -> str:
    data = {
        "id": session_id,
        "start_date": start_date.strftime("%m-%d-%Y-%H-%M-%S"),
    }

    dumped = json.dumps(data)
    return base64.urlsafe_b64encode(dumped.encode("utf-8")).decode("ascii")


app.mount(
    "/static",
    staticfiles.StaticFiles(
        directory=Path(__file__).parent.absolute().joinpath("static"), html=True
    ),
    name="static",
)


class SpkInitData(BaseModel):
    speaker_name: str


class MeetingInitData(BaseModel):
    meeting_name: str
    meeting_speakers: List[str]


@app.get("/")
def root_page(req: Request):
    return templates.TemplateResponse("index.html", {"request": req})


@app.get("/record/speaker")
def speakers_page(req: Request):
    return templates.TemplateResponse("record_speaker.html", {"request": req})


@app.get("/record/meeting")
async def meetings_page(req: Request):
    speaker_items = await speakers_col.find({}, {"speaker_id": 1}).to_list(length=None)
    return templates.TemplateResponse(
        "record_meeting.html", {"request": req, "speaker_items": speaker_items}
    )


@app.get("/export")
async def export_page(
    req: Request,
    page: int = Query(default=1, gt=0),
    page_size: int = Query(default=8, gt=0),
):
    metadata_headers = [
        "meeting_id",
        "start_date",
        "end_date",
        "recognized_speakers",
        "selected_speakers",
        "recognized_speakers_ratio",
    ]

    sessions_cnt = await sessions_col.count_documents({"status": "completed"})
    meetings_data = (
        await sessions_col.find({"status": "completed"}, {"data": 0})
        .sort("start_date", -1)
        .skip((page - 1) * page_size)
        .limit(page_size)
        .to_list(length=None)
    )

    for m in meetings_data:
        m["recognized_speakers"] = "|".join(m["recognized_speakers"])
        m["selected_speakers"] = "|".join(m["selected_speakers"])

    page_limit = math.ceil(sessions_cnt / page_size)

    page_data = {"current_page": page}
    if page > 1:
        page_data["page_prev"] = page - 1
    if page < page_limit:
        page_data["page_next"] = page + 1

    return templates.TemplateResponse(
        "export_meetings.html",
        {
            "request": req,
            "headers": metadata_headers,
            "rows": meetings_data,
            "page_data": page_data,
        },
    )


@app.post("/spk/init")
async def create_spk_session(data: SpkInitData):
    speaker_id = data.speaker_name
    start_date = datetime.now()
    spk_session_cookie = create_session_cookie(speaker_id, start_date)

    await speakers_col.update_one(
        {"speaker_id": speaker_id},
        {
            "$set": {
                "speaker_name": speaker_id,
                "start_date": start_date,
                "status": "in_progress",
                "data": [],
            }
        },
        upsert=True,
    )

    resp = JSONResponse(
        status_code=status.HTTP_201_CREATED, content={"status": "initiated"}
    )
    resp.set_cookie(key="spk_session", value=spk_session_cookie)

    return resp


@app.get("/spk/finish")
async def create_spk_session(
    spk_data: dict = Depends(get_spk_cookie_data_http),
):
    speaker_id = spk_data["id"]
    end_date = datetime.now()

    stats = await speakers_col.aggregate(
        [
            {"$match": {"speaker_id": speaker_id}},
            {
                "$project": {
                    "_id": "$speaker_id",
                    "good_vectors_num": {
                        "$size": {
                            "$filter": {
                                "input": "$data",
                                "as": "elem",
                                "cond": {
                                    "$gte": ["$$elem.spk_frames", GOOD_SPK_FRAMES_NUM]
                                },
                            }
                        }
                    },
                    "all_vectors_num": {"$size": "$data"},
                }
            },
        ]
    ).to_list(length=1)

    stats = stats[0]

    all_vectors_num = stats["all_vectors_num"]
    spk_ratio = 0.0

    if all_vectors_num > 0:
        spk_ratio = stats["good_vectors_num"] / all_vectors_num

    await speakers_col.update_one(
        {"speaker_id": speaker_id},
        {
            "$set": {
                "end_date": end_date,
                "status": "completed",
                "spk_ratio": spk_ratio,
            }
        },
    )

    msg = []
    verdict = True

    if all_vectors_num < MIN_SPK_VECTORS_NUM:
        verdict = verdict and False
        msg += ["for a longer time"]

    if spk_ratio < SPK_GOOD_RATIO:
        verdict = verdict and False
        msg += ["in longer sentences"]

    if verdict == False:
        msg = "Please, try to speak " + ",".join(msg)
    else:
        msg = "Good speaker recording"

    resp = JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "finished",
            "stats": {
                "verdict": verdict,
                "msg": msg,
                "spk_ratio": spk_ratio,
                "vectors_num": all_vectors_num,
                "vectors_num_req": MIN_SPK_VECTORS_NUM,
            },
        },
    )

    resp.delete_cookie(key="spk_session")
    return resp


@app.post("/meeting/init")
async def create_recog_session(meeting_data: MeetingInitData):
    meeting_name = meeting_data.meeting_name
    speakers = meeting_data.meeting_speakers

    start_date = datetime.now()
    meeting_id = f"{meeting_name}_{start_date.strftime('%m-%d-%Y-%H-%M-%S')}"
    meeting_session_cookie = create_session_cookie(meeting_id, start_date)

    await sessions_col.update_one(
        {"meeting_id": meeting_id},
        {
            "$set": {
                "meeting_id": meeting_id,
                "recognized_speakers": [],
                "selected_speakers": speakers,
                "start_date": start_date,
                "status": "in_progress",
                "data": [],
            }
        },
        upsert=True,
    )

    resp = JSONResponse(
        status_code=status.HTTP_201_CREATED, content={"status": "initiated"}
    )
    resp.set_cookie(key="meeting_session", value=meeting_session_cookie)
    return resp


@app.get("/meeting/finish")
async def finish_recog_session(
    meeting_data: dict = Depends(get_meeting_cookie_data_http),
):
    meeting_id = meeting_data["id"]
    end_date = datetime.now()

    speakers_data = await sessions_col.find_one(
        {"meeting_id": meeting_id}, {"selected_speakers": 1, "recognized_speakers": 1}
    )

    t_speakers = speakers_data["selected_speakers"]
    r_speakers = speakers_data["recognized_speakers"]

    t_len = len(speakers_data["selected_speakers"])
    r_len = t_len - len(set(t_speakers).difference(r_speakers))
    speakers_ratio = r_len / t_len

    await sessions_col.update_one(
        {"meeting_id": meeting_id},
        {
            "$set": {
                "end_date": end_date,
                "status": "completed",
                "recognized_speakers_ratio": speakers_ratio,
            }
        },
    )

    resp = JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "finished",
            "meeting_id": meeting_id,
            "stats": {
                "speakers_all_num": t_len,
                "speakers_recognized_num": r_len,
            },
        },
    )

    resp.delete_cookie(key="meeting_session")
    return resp


@app.websocket("/spk/ws/")
async def spk_ws(
    socket: WebSocket,
    speaker_data: dict = Depends(get_spk_cookie_data_ws),
):
    speaker_id = speaker_data["id"]

    config = {"config": {"sample_rate": 16000}}

    await socket.accept()

    async with websockets.connect(VOSK_SERVER_WS_URL) as vosk_server:
        await vosk_server.send(json.dumps(config))
        try:
            while True:
                data = await socket.receive_bytes()
                await vosk_server.send(data)
                raw_data = await vosk_server.recv()
                data = json.loads(raw_data)

                if spk_res := extract_spk_from_result(data):
                    await speakers_col.update_one(
                        {"speaker_id": speaker_id},
                        {"$push": {"data": spk_res._asdict()}},
                    )

        except WebSocketDisconnect:
            await vosk_server.send('{"eof" : 1}')
            final_raw_data = await vosk_server.recv()
            final_data = json.loads(final_raw_data)

            if spk_res := extract_spk_from_result(final_data):
                await speakers_col.update_one(
                    {"speaker_id": speaker_id},
                    {"$push": {"data": spk_res._asdict()}},
                )


@app.websocket("/meeting/ws/")
async def recognize_ws(
    socket: WebSocket, meeting_data: dict = Depends(get_meeting_cookie_data_ws)
):
    await socket.accept()

    loop = asyncio.get_running_loop()
    meeting_id = meeting_data["id"]
    meeting = await sessions_col.find_one(
        {"status": "in_progress", "meeting_id": meeting_id}
    )
    spk_vectors = await speakers_col.aggregate(
        [
            {
                "$match": {
                    "speaker_id": {"$in": meeting["selected_speakers"]},
                    "status": "completed",
                }
            },
            {"$project": {"speaker_id": "$speaker_id", "data": "$data.spk"}},
        ]
    ).to_list(length=None)

    spk_vectors = {el["speaker_id"]: el["data"] for el in spk_vectors}
    process_fn = partial(process_vosk_result, spk_vectors=spk_vectors)
    config = {"config": {"sample_rate": 16000}}

    async with websockets.connect(VOSK_SERVER_WS_URL) as vosk_server:
        await vosk_server.send(json.dumps(config))

        await socket.send_json(
            {
                "type": "meeting_setup_done",
                "status_msg": f"loaded {len(spk_vectors)} speakers, vosk_server connected",
            }
        )

        try:
            while True:
                data = await socket.receive_bytes()
                await vosk_server.send(data)
                raw_data = await vosk_server.recv()
                data = json.loads(raw_data)

                vosk_res = await loop.run_in_executor(None, process_fn, data)
                if vosk_res:
                    await sessions_col.update_one(
                        {"meeting_id": meeting_id},
                        {
                            "$push": {"data": vosk_res._asdict()},
                            "$addToSet": {"recognized_speakers": vosk_res.speaker},
                        },
                    )

                    await socket.send_json(
                        {
                            "type": "chunk_processed",
                            "speaker": vosk_res.speaker,
                            "text": vosk_res.text,
                        }
                    )

        except WebSocketDisconnect:
            await vosk_server.send('{"eof" : 1}')
            final_raw_data = await vosk_server.recv()
            final_data = json.loads(final_raw_data)

            vosk_res = await loop.run_in_executor(None, process_fn, final_data)
            if vosk_res:
                await sessions_col.update_one(
                    {"meeting_id": meeting_id},
                    {
                        "$push": {"data": vosk_res._asdict()},
                        "$addToSet": {"recognized_speakers": vosk_res.speaker},
                    },
                )


def is_close(a, b):
    speakers_eq = a["speaker"] == b["speaker"]
    time_diff_eq = abs(b["start"] - a["end"]) < MERGE_DIFF_SEC
    return speakers_eq and time_diff_eq


def merge(a, b):
    return {
        "start": a["start"],
        "end": b["end"],
        "text": " ".join((a["text"], b["text"])).strip(),
        "conf": (a["conf"] + b["conf"]) / 2,
        "speaker": a["speaker"],
        "is_question": a["is_question"],
    }


async def process_export_meeting(meeting_id: str):
    meeting_records = sessions_col.aggregate(
        [
            {"$match": {"meeting_id": meeting_id, "status": "completed"}},
            {"$unwind": "$data"},
            {"$replaceRoot": {"newRoot": "$data"}},
            {"$sort": {"start": 1}},
        ]
    )

    questions = []
    q_phrases = 0
    async for record in meeting_records:
        if not questions or not is_close(questions[-1], record):
            if record["is_question"]:
                questions.append(record)
                q_phrases = 0

        elif q_phrases <= 5:
            questions[-1] = merge(questions[-1], record)
            q_phrases += 1

    return questions


def create_csv_content(headers, rows) -> str:
    with StringIO() as s:
        writer = csv.DictWriter(
            s, delimiter=",", fieldnames=headers, extrasaction="ignore"
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        return s.getvalue()


def zipfiles(zip_name: str, files: dict[str, Union[str, StringIO]]):
    zip_filename = f"{zip_name}.zip"

    # Open StringIO to grab in-memory ZIP contents
    zip_io = BytesIO()
    # The zip compressor
    with zipfile.ZipFile(zip_io, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)

    # Grab ZIP file from in-memory, make response with correct MIME-type
    return StreamingResponse(
        iter([zip_io.getvalue()]),
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"},
    )


@app.get("/meeting/export/{meeting_id}")
async def export_meeting(meeting_id: str):
    meeting_data = await sessions_col.find_one(
        {"meeting_id": meeting_id, "status": "completed"}, {"data": 0}
    )

    metadata_headers = [
        "meeting_id",
        "start_date",
        "end_date",
        "recognized_speakers",
        "selected_speakers",
        "recognized_speakers_ratio",
    ]

    meeting_data["recognized_speakers"] = "|".join(meeting_data["recognized_speakers"])
    meeting_data["selected_speakers"] = "|".join(meeting_data["selected_speakers"])
    metadata_csv_io = create_csv_content(metadata_headers, [meeting_data])

    meeting_questions = await process_export_meeting(meeting_id)
    questions_csv_headers = ["speaker", "text", "start", "end", "conf"]

    questions_csv_io = create_csv_content(questions_csv_headers, meeting_questions)

    return zipfiles(
        meeting_id,
        {
            "metadata.csv": metadata_csv_io,
            "questions.csv": questions_csv_io,
        },
    )
