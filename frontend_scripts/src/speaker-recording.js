const { MicrophoneStream }= require('microphone-stream');
const { convertFloat32ArrayToInt16Array }= require('./float32-to-int16');

const startRecording = document.getElementById('start-recording');
const stopRecording = document.getElementById('stop-recording');
const resetRecording = document.getElementById('reset-recording');
const speakerForm = document.getElementById('set-speaker');

const speakerSetStatus = document.getElementById('speaker-set-status');
const speakerRecordingStatus = document.getElementById('speaker-recording-status');
const recordingStats = document.getElementById('speaker-recording-stats');

let ws;


resetRecording.onclick = function () {
  window.location.reload();
  return false;
};

speakerForm.onsubmit = function (event) {
  event.preventDefault();
  event.stopPropagation();

  const formData = new FormData(speakerForm);

  if (speakerForm.querySelector('input[name="speaker_name"]').value == "") {
    alert("Speaker name cannot be empty");
    return false;
  }

  fetch("/spk/init", {
    method: "POST",
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(Object.fromEntries(formData.entries()))
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
        speakerForm.querySelector('input[type="submit"]').disabled = true;
        speakerSetStatus.className = "speaker-set";
        speakerSetStatus.querySelector("p").innerText = "Speaker is set";
        startRecording.disabled = false;
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

  ws = new WebSocket(((window.location.protocol === "https:") ? "wss://" : "ws://") + window.location.host + "/spk/ws/");

  navigator.mediaDevices.getUserMedia({ video: false, audio: true })
    .then(function (stream) {
      micStream.setStream(stream);

      timerId = setInterval(() => {
        recordingTime += 1;
        speakerRecordingStatus.querySelector("p").innerText = `Recording in progress: ${parseInt(recordingTime / 60)}m ${recordingTime % 60}s`;
      }, 1000);

      startRecording.disabled = true;
      stopRecording.disabled = false;

      speakerRecordingStatus.className = "speaker-set";

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
    ws.close();
    micStream.stop();
    resetRecording.hidden = false;
    resetRecording.disabled = false;
    stopRecording.disabled = true;

    setTimeout(() => {
      clearInterval(timerId);
      speakerRecordingStatus.querySelector("p").innerText = `Recording finished, total time: ${parseInt(recordingTime / 60)}m ${recordingTime % 60}s`;

      fetch("/spk/finish", {
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
            const stats = respJson.stats;
            recordingStats.hidden = false;

            if (stats.verdict == true) {
              recordingStats.className = "speaker-set";
            } else {
              recordingStats.className = "speaker-not-set";
            }

            recordingStats.querySelector("p").innerText =
              `Stats: vectors recorded - ${stats.vectors_num}/${stats.vectors_num_req}, `
              + `vectors quality ratio - ${stats.spk_ratio}, `
              + `verdict: ${stats.msg}`
          }
        })
        .catch((err) => {
          alert(err);
        });
    }, 1000);
  };
}
