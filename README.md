# zoom-assistant

Web application built with `Python3.10` and `Vosk` for transcribing audio from microphone on the fly and detecting questions (Russian language), useful for meetings.

**Features**:
- recording speakers audio data and preparing speakers pool
- online microphone stream processing with transcription and speaker detection
- export of finished audio sessions (metadata + detected questions)

## Pre-requirements

Docker, x86_64/amd64 architecture (only for vosk server, doesn't support arm)

## Running web app

### env vars

* `VOSK_SERVER_WS_URL` - url where vosk websocket server is started, default is `ws://localhost:2700`
* `MONGODB_URI` - url to mongodb, default is `mongodb://localhost:27017`
* `MONGO_VOSK_DB_NAME` - name of mongo database, default is `nir-zoom`
* `MONGO_SPEAKERS_COL_NAME` - name for collection with speaker records, default is `speakers`
* `MONGO_SESSIONS_COL_NAME` - name for collection with meetings records, default is `sessions`
* `GOOD_SPK_FRAMES_NUM` - value for analyzing quality of recorded speaker features, default is `300`
* `MIN_SPK_VECTORS_NUM` - value for checking quantity of recorded speaker features, default is `8`
* `SPK_GOOD_RATIO` - value restricting min border of ratio = (num good speaker features / num all speaker features), default is `0.65`
* `MERGE_DIFF_SEC` - value in secs - how close should be two phrases to be combined into one, default is `2.5` s

### starting procedures

_TODO_

### endpoint docs

Available at `GET /docs` endpoint, navigate using browser

## Using web app

### Main page

<img width="1065" alt="image" src="https://user-images.githubusercontent.com/31539612/168853694-fc80e8db-d50b-4b69-9296-20ae9ab7fa1a.png">

- To record speaker, click on the `Record speakers` link
- To record meeting session, click on the `Record meeting` link
- To view and export recorded meetings, click on the `Export meeting stats` link

### Speaker recording

<img width="1065" alt="image" src="https://user-images.githubusercontent.com/31539612/168854241-c6a6ffac-0463-4b92-b089-39aa1a97ba97.png">

1. Input speaker name in the form and click `Set speaker` button, app would initialize a speaker recording session, and recording button will become available. Name cannot be changed, if you made a typo you would need to reload page and create new speaker
2. Click on the `Start recording` button and allow microphone usage in the following broweser prompt
3. Wait for 1 sec and start speaking, loud and clear. Try to speak in long sentences, with pauses between sentences for about 2 seconds. Total speaking time should be about 1 - 2 minutes
4. When you're done, click on the `Stop recording` button. App will analyze recorded data and display total time, along with quality and quantity of recorded data. If recording is not quite good, there will be recommendations for improvment
5. Recording complete, if you wish to record another speaker, simply reload the page or click `Record another speaker` button (will appear after recording is stopped).

> if you want to record different data for the same speaker, input the same name in next recording session, data for previous recording would be overwritten.

### Audio session recording

<img width="1035" alt="image" src="https://user-images.githubusercontent.com/31539612/168857336-524cdf24-48f3-4881-a125-e6cc8dae6221.png">


1. Input meeting name in form and select speakers from dropdown multiselect with checkboxes. After click on the `Set meeting data` button to initilize a meeting session.
2. Click `Start recording` and allow microphone usage in the following broweser prompt, then after 1 sec app would start processing microphone audio data on the fly and display detected phrases with speaker name in the appeared textarea below buttons.
3. When you wish to stop processing and finish recording microphone data, click `Stop recording` button. App will analyze recorded data and display total time, along with info about how many speakers have been actually detected (from those you've chosen at the start)
4. The `Export` button will appear, if you wish to export meeting stats right away. If you don't, you can always view and export stats at export page.
5. Recorging is done, you can start another audio session or close the page

> App will add a timestamp to each session name, so same names can be used for different sessions

> App will use the microphone which is being used by the browser, so if you wish to change micro, you will have to do it in browser or OS settings

### View and export recorded sessions

<img width="1440" alt="image" src="https://user-images.githubusercontent.com/31539612/168857173-0908d954-7884-436b-9ce6-daa4119b689e.png">

Page support pagination, page shows 8 records, when sessions number will be over 8, the controls for navigating between different pages will appear around `Page N` text.

To export any record, just click `Export`, data will be exported in zip archive with 2 csv files - `metadata.csv` will contain data shown in table, `questions.csv` will contain speakers questionta daa with timestamps in seconds relative to start of the recording

## Code notes

### Vosk utils

1. Speaker detection is based on `cosine distance` between detected speaker features vector (using Vosk) and pre-recorded features vectors - for each speaker set the mean distance is calculated and speaker with minimal mean distance is being picked
2. Question detecting is rule-based, rules are gathered in `QUESTION_RULES` array at the top of `vosk_utils/__init__.py` file, each rule is the boolen function, with checks keywords entry in passed text
3. Forbidden ("bad") words are being stored as pickled python list in `bad_words.pkl` in `vosk_utils` folder, if you wish to extend this list, you need to unpicke this file, extend python list with needed words and pickle it again. Bad words are being simply erased for recognized phrases
4. Questions during export are being calculated in following algo: if rule-based algo determined the phrase is a question, at most 5 next subsequent phrases from same speaker (without interruption from other speakers) would be added to the question text. The logic behind this is that rule-based algo looking to the start of the question, but following phrases even after a pauses are usually still a part of the question, even if they don't contain explicit question keywords

### CLI app

Use `python3.10 main.py --help` to invoke detailed description:

<img width="1013" alt="image" src="https://user-images.githubusercontent.com/31539612/168869347-f4de4580-9126-4697-b69c-283525435477.png">

## outdated info
Link to screencast with first results (19.03.2022 - cli app) - https://drive.google.com/file/d/1MQdnaoQoiWK9L1MZP185QtTg8CifSfz9/view?usp=sharing
