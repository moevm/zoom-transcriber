FROM python:3.10-slim-buster

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install -U pip setuptools && pip install -r /app/requirements.txt

COPY ./backend /app/backend
COPY ./vosk_utils /app/vosk_utils

CMD gunicorn backend:app --bind=0.0.0.0:3030 -w 4 -k uvicorn.workers.UvicornWorker
