const MicrophoneStream = require('microphone-stream').default;
const { convertFloat32ArrayToInt16Array } = require("./float32-to-int16");

const startRecording = document.getElementById('start-recording');
const stopRecording = document.getElementById('stop-recording');
const resetRecording = document.getElementById('reset-recording');
const meetingForm = document.getElementById('set-meeting');

const meetingSetStatus = document.getElementById('meeting-set-status');
const meetingRecordingStatus = document.getElementById('meeting-recording-status');
const recordingStats = document.getElementById('meeting-recording-stats');
const recognizedText = document.getElementById('recognized-text');
const recognizedTextArea = document.getElementById('recognized-text-area');
const mettingBts = document.getElementById('meetings-buttons');

let ws;

function parseWords(words) {
  return words
    .map((w) => {
      console.log(w);
      let conf = w.conf;
      let text = w.word;
      return conf > 0.7 ? text : `<span class=bad-word>${text}</span>`
    })
    .join('&nbsp;')
}


resetRecording.onclick = function () {
  window.location.reload();
  return false;
};

meetingForm.onsubmit = function (event) {
  event.preventDefault();
  event.stopPropagation();

  const formData = new FormData(meetingForm);

  if (meetingForm.querySelector('input[name="meeting_name"]').value == "") {
    alert("Meeting name cannot be empty");
    return false;
  }

  var jsonMeetingData = {};

  const meeting_name = meetingForm.querySelector("input[name='meeting_name']").value;
  const speakerOptions = meetingForm.querySelectorAll("select option");

  jsonMeetingData["meeting_name"] = meeting_name;
  const speakers = [];
  for (sp of speakerOptions) {
    if (sp.selected) {
      speakers.push(sp.value);
    }
  }

  jsonMeetingData["meeting_speakers"] = speakers;

  fetch("/meeting/init", {
    method: "POST",
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(jsonMeetingData)
  })
    .then((response) => {
      if (!response.ok) {
        console.log(response);
        throw new Error(`ERROR: session has not been initiated with status: ${response.status}`);
      }
      return response.json();
    })
    .then((respJson) => {
      if (respJson["status"] == "initiated") {
        ws = new WebSocket(((window.location.protocol === "https:") ? "wss://" : "ws://") + window.location.host + "/meeting/ws/");

        ws.onmessage = (e) => {
          const msg = JSON.parse(e.data);
          switch (msg.type) {
            case "meeting_setup_done":
              meetingForm.querySelector('input[type="submit"]').disabled = true;
              meetingSetStatus.className = "speaker-set";
              meetingSetStatus.querySelector("p").innerText = `Meeting is set: ${msg.status_msg}`;
              startRecording.disabled = false;
              recognizedText.hidden = false;
              recognizedText.className = "col-container";
              break;
            case "chunk_processed":
              recognizedText.innerHTML += `<p>(${msg.start}) ${msg.speaker}: ${msg.words.length > 0 ? parseWords(msg.words) : msg.text}</p>`
              break;
          }
        }
      }
    })
    .catch((err) => {
      alert(err);
    })
}


startRecording.onclick = function () {


  let recordingTime = 0;


  let timerId;

  // Note: in most browsers, this constructor must be called in response to a click/tap,
  // or else the AudioContext will remain suspended and will not provide any audio data.
  const micStream = new MicrophoneStream({
    context: new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 })
  });



  navigator.mediaDevices.getUserMedia({ video: false, audio: true })
    .then(function (stream) {
      micStream.setStream(stream);

      timerId = setInterval(() => {
        recordingTime += 1;
        meetingRecordingStatus.querySelector("p").innerText = `Recording in progress: ${parseInt(recordingTime / 60)}m ${recordingTime % 60}s`;
      }, 1000);

      startRecording.disabled = true;
      stopRecording.disabled = false;

      meetingRecordingStatus.className = "speaker-set";

    }).catch(function (error) {
      console.log(error);
    });

  // get Buffers (Essentially a Uint8Array DataView of the same Float32 values)
  micStream.on('data', function (chunk) {
    // Optionally convert the Buffer back into a Float32Array
    // (This actually just creates a new DataView - the underlying audio data is not copied or modified.)

    const rawChunk = MicrophoneStream.toRaw(chunk)
    ws.send(convertFloat32ArrayToInt16Array(rawChunk));

    // note: if you set options.objectMode=true, the `data` event will output AudioBuffers instead of Buffers
  });

  // It also emits a format event with various details (frequency, channels, etc)
  micStream.on('format', function (format) {
    console.log(format);
  });

  // Stop when ready
  stopRecording.onclick = function () {
    if (ws) ws.close();
    micStream.stop();
    resetRecording.hidden = false;
    resetRecording.disabled = false;
    stopRecording.disabled = true;

    setTimeout(() => {
      clearInterval(timerId);
      meetingRecordingStatus.querySelector("p").innerText = `Recording finished, total time: ${parseInt(recordingTime / 60)}m ${recordingTime % 60}s`;

      fetch("/meeting/finish", {
        method: "GET",
        credentials: 'same-origin',
      })
        .then((response) => {
          if (!response.ok) {
            console.log(response);
            throw new Error(`ERROR: session has not been finished with status: ${response.status}`);
          }
          return response.json()
        })
        .then((respJson) => {
          if (respJson.status == "finished") {
            const meetingId = respJson.meeting_id;
            const stats = respJson.stats;
            recordingStats.hidden = false;

            recordingStats.className = "speaker-set";

            recordingStats.querySelector("p").innerText =
              `Stats: speakers recognized - ${stats.speakers_recognized_num}/${stats.speakers_all_num}`;


            exportBtn = document.createElement('a');
            exportBtn.innerText = 'Export meeting';
            exportBtn.setAttribute('href', `/meeting/export/${meetingId}`);
            exportBtn.setAttribute('download', 'download');
            exportBtn.className = "link-btn";
            mettingBts.appendChild(exportBtn);
          }
        })
        .catch((err) => {
          alert(err);
        });
    }, 1000);
  };
}
