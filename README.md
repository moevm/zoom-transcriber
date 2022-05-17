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
_TODO_

### starting procedures
_TODO_

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


# outdated info
Link to screencast with first results (19.03.2022 - cli app) - https://drive.google.com/file/d/1MQdnaoQoiWK9L1MZP185QtTg8CifSfz9/view?usp=sharing
