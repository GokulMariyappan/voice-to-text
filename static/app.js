const startRecordButton = document.getElementById("startRecord");
const stopRecordButton = document.getElementById("stopRecord");
const transcribeRecordingButton = document.getElementById("transcribeRecording");
const transcribeUploadButton = document.getElementById("transcribeUpload");
const analyzeTextButton = document.getElementById("analyzeText");
const recordingStatus = document.getElementById("recordingStatus");
const apiStatus = document.getElementById("apiStatus");
const transcriptField = document.getElementById("transcript");
const audioFileInput = document.getElementById("audioFile");
const recordingPreview = document.getElementById("recordingPreview");
const symptomsOutput = document.getElementById("symptomsOutput");
const analysisOutput = document.getElementById("analysisOutput");
const sourcesOutput = document.getElementById("sourcesOutput");

let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;
let activeStream = null;

function setStatus(element, message, isError = false) {
  element.textContent = message;
  element.classList.toggle("error", isError);
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function isSecureEnoughForMic() {
  return window.isSecureContext || ["localhost", "127.0.0.1"].includes(window.location.hostname);
}

function getSupportedMimeType() {
  if (typeof MediaRecorder === "undefined") {
    return "";
  }

  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];

  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function describeMicError(error) {
  if (!error) return "Microphone access failed.";
  if (error.name === "NotAllowedError") return "Microphone permission denied.";
  if (error.name === "NotFoundError") return "No microphone found.";
  if (error.name === "NotReadableError") return "Microphone is busy.";
  if (error.name === "SecurityError") return "Requires HTTPS or localhost.";
  return error.message || "Microphone access failed.";
}

function stopActiveStream() {
  if (activeStream) {
    activeStream.getTracks().forEach((track) => track.stop());
    activeStream = null;
  }
}

async function transcribeBlob(blob, filename) {
  const formData = new FormData();
  formData.append("audio", blob, filename);
  setStatus(apiStatus, "Transcribing audio...");

  const response = await fetch("/api/transcribe", {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.error || "Transcription failed.");
  }

  transcriptField.value = payload.transcript || "";
  setStatus(apiStatus, "Transcription complete.");
}

startRecordButton.addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    setStatus(recordingStatus, "Microphone recording not supported.", true);
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    activeStream = stream;
    audioChunks = [];
    recordedBlob = null;
    recordingPreview.hidden = true;
    recordingPreview.src = "";

    const mimeType = getSupportedMimeType();
    const options = mimeType ? { mimeType } : {};
    mediaRecorder = new MediaRecorder(stream, options);

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    });

    mediaRecorder.addEventListener("stop", () => {
      recordedBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
      transcribeRecordingButton.disabled = false;
      recordingPreview.src = URL.createObjectURL(recordedBlob);
      recordingPreview.hidden = false;
      setStatus(recordingStatus, "Recording ready.");
      stopActiveStream();
    });

    mediaRecorder.addEventListener("error", (event) => {
      setStatus(recordingStatus, `Error: ${event.error?.message || "unknown"}`, true);
      stopActiveStream();
    });

    mediaRecorder.start();
    startRecordButton.disabled = true;
    stopRecordButton.disabled = false;
    transcribeRecordingButton.disabled = true;
    setStatus(recordingStatus, "Recording...");
  } catch (error) {
    setStatus(recordingStatus, describeMicError(error), true);
    stopActiveStream();
  }
});

stopRecordButton.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    startRecordButton.disabled = false;
    stopRecordButton.disabled = true;
  }
});

transcribeRecordingButton.addEventListener("click", async () => {
  if (!recordedBlob) return;
  try {
    await transcribeBlob(recordedBlob, "recording.webm");
  } catch (error) {
    setStatus(apiStatus, error.message, true);
  }
});

transcribeUploadButton.addEventListener("click", async () => {
  const file = audioFileInput.files[0];
  if (!file) {
    setStatus(apiStatus, "Select a file first.", true);
    return;
  }
  try {
    await transcribeBlob(file, file.name);
  } catch (error) {
    setStatus(apiStatus, error.message, true);
  }
});

analyzeTextButton.addEventListener("click", async () => {
  const text = transcriptField.value.trim();
  if (!text) {
    setStatus(apiStatus, "No text to analyze.", true);
    return;
  }

  try {
    setStatus(apiStatus, "Analyzing...");
    symptomsOutput.textContent = "Loading...";
    analysisOutput.textContent = "Loading...";
    sourcesOutput.innerHTML = "Loading...";

    console.log("Starting analysis for text:", text);
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const payload = await response.json();
    console.log("Analysis payload received:", payload);

    if (!response.ok) throw new Error(payload.error || "Analysis failed.");

    symptomsOutput.textContent = pretty(payload.symptoms);
    analysisOutput.textContent = pretty(payload.analysis);

    const cards = [];
    (payload.evidence.pubmed || []).forEach(article => {
      cards.push(`
        <div class="source-card">
          <strong>PubMed Reference</strong>
          <div>${article.title}</div>
          <a href="${article.url}" target="_blank">View Source</a>
        </div>
      `);
    });

    (payload.evidence.validated_conditions || []).forEach(v => {
      cards.push(`
        <div class="source-card">
          <strong>${v.candidate_name}</strong>
          <div>${v.medlineplus?.title || "NLM Record"}</div>
          ${(v.medlineplus?.links || []).map(l => `<a href="${l.href}" target="_blank">MedlinePlus</a>`).join("")}
        </div>
      `);
    });

    sourcesOutput.innerHTML = cards.length ? cards.join("") : "No sources found.";
    setStatus(apiStatus, "Analysis complete.");
  } catch (error) {
    setStatus(apiStatus, error.message, true);
  }
});
