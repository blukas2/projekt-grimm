(function () {
  if (window.projektGrimmLive) {
    return;
  }

  const SPEECH_THRESHOLD = 0.02;
  const SILENCE_DURATION_MS = 900;

  const state = {
    captureContext: null,
    outputContext: null,
    microphoneStream: null,
    processor: null,
    source: null,
    silenceNode: null,
    socket: null,
    playbackTime: 0,
    stopping: false,
    speechActive: false,
    silenceMs: 0,
    transcriptSpeaker: null,
    transcriptTextNode: null,
  };

  async function startConversation() {
    if (state.socket) {
      return;
    }
    setStatus('connecting', 'Verbinde mit Gemini Live ...');
    clearTranscript();
    appendTranscript('System', 'Mikrofon wird vorbereitet.');
    try {
      await prepareCapture();
      await preparePlayback();
      connectSocket();
    } catch (error) {
      cleanupAudio();
      setStatus('error', formatError(error));
      appendTranscript('System', formatError(error));
    }
  }

  function stopConversation() {
    state.stopping = true;
    setStatus('stopping', 'Live-Unterhaltung wird beendet ...');
    sendActivityEnd();
    if (state.socket && state.socket.readyState === WebSocket.OPEN) {
      state.socket.send('stop');
    }
    cleanupAudio();
    if (state.socket) {
      state.socket.close();
      state.socket = null;
    }
    setStatus('inactive', 'Noch keine Live-Unterhaltung aktiv.');
  }

  async function prepareCapture() {
    state.captureContext = new AudioContext({ sampleRate: 16000 });
    state.microphoneStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
    state.source = state.captureContext.createMediaStreamSource(state.microphoneStream);
    state.processor = state.captureContext.createScriptProcessor(4096, 1, 1);
    state.silenceNode = state.captureContext.createGain();
    state.silenceNode.gain.value = 0;
    state.processor.onaudioprocess = sendAudioChunk;
    state.source.connect(state.processor);
    state.processor.connect(state.silenceNode);
    state.silenceNode.connect(state.captureContext.destination);
  }

  async function preparePlayback() {
    state.outputContext = new AudioContext({ sampleRate: 24000 });
    state.playbackTime = state.outputContext.currentTime;
    if (state.outputContext.state === 'suspended') {
      await state.outputContext.resume();
    }
  }

  function connectSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    state.socket = new WebSocket(`${protocol}://${window.location.host}/api/live-audio`);
    state.socket.binaryType = 'arraybuffer';
    state.socket.addEventListener('message', handleSocketMessage);
    state.socket.addEventListener('close', handleSocketClose);
    state.socket.addEventListener('error', handleSocketError);
  }

  function sendAudioChunk(event) {
    if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    const chunkDurationMs = chunkDuration(input.length);
    if (hasSpeech(input)) {
      sendActivityStart();
      state.silenceMs = 0;
      sendPcmChunk(input);
      return;
    }
    if (!state.speechActive) {
      return;
    }
    sendPcmChunk(input);
    state.silenceMs += chunkDurationMs;
    if (state.silenceMs >= SILENCE_DURATION_MS) {
      sendActivityEnd();
    }
  }

  function handleSocketMessage(event) {
    if (typeof event.data === 'string') {
      handleJsonMessage(JSON.parse(event.data));
      return;
    }
    playAudioChunk(event.data);
  }

  function handleJsonMessage(message) {
    if (message.type === 'status') {
      setStatus(message.state, message.message);
      appendTranscript('System', message.message);
      return;
    }
    if (message.type === 'transcript') {
      const speaker = message.speaker === 'user' ? 'Du' : 'Lehrer';
      appendTranscriptChunk(speaker, message.text);
      return;
    }
    if (message.type === 'error') {
      setStatus('error', message.message);
      appendTranscript('System', message.message);
    }
  }

  function handleSocketClose() {
    cleanupAudio();
    state.socket = null;
    if (state.stopping) {
      state.stopping = false;
      setStatus('inactive', 'Noch keine Live-Unterhaltung aktiv.');
      return;
    }
    setStatus('inactive', 'Live-Unterhaltung beendet.');
  }

  function handleSocketError() {
    setStatus('error', 'Die Live-Verbindung ist fehlgeschlagen.');
  }

  function playAudioChunk(arrayBuffer) {
    if (!state.outputContext) {
      return;
    }
    const pcmData = new Int16Array(arrayBuffer);
    const audioBuffer = state.outputContext.createBuffer(1, pcmData.length, 24000);
    const channelData = audioBuffer.getChannelData(0);
    for (let index = 0; index < pcmData.length; index += 1) {
      channelData[index] = pcmData[index] / 32768;
    }
    const source = state.outputContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(state.outputContext.destination);
    const startTime = Math.max(state.outputContext.currentTime, state.playbackTime);
    source.start(startTime);
    state.playbackTime = startTime + audioBuffer.duration;
  }

  function cleanupAudio() {
    state.speechActive = false;
    state.silenceMs = 0;
    disconnectNode(state.source);
    disconnectNode(state.processor);
    disconnectNode(state.silenceNode);
    stopTracks();
    closeContext('captureContext');
    closeContext('outputContext');
    state.source = null;
    state.processor = null;
    state.silenceNode = null;
    state.microphoneStream = null;
  }

  function disconnectNode(node) {
    if (!node) {
      return;
    }
    try {
      node.disconnect();
    } catch (error) {
      console.debug('Node disconnect skipped', error);
    }
  }

  function stopTracks() {
    if (!state.microphoneStream) {
      return;
    }
    state.microphoneStream.getTracks().forEach(track => track.stop());
  }

  function closeContext(key) {
    const context = state[key];
    if (!context) {
      return;
    }
    context.close().catch(() => null);
    state[key] = null;
  }

  function toPcm16(floatData) {
    const pcm = new ArrayBuffer(floatData.length * 2);
    const view = new DataView(pcm);
    for (let index = 0; index < floatData.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, floatData[index]));
      view.setInt16(index * 2, sample * 32767, true);
    }
    return pcm;
  }

  function sendPcmChunk(floatData) {
    const pcmChunk = toPcm16(floatData);
    state.socket.send(pcmChunk);
  }

  function hasSpeech(floatData) {
    let sum = 0;
    for (let index = 0; index < floatData.length; index += 1) {
      sum += floatData[index] * floatData[index];
    }
    return Math.sqrt(sum / floatData.length) >= SPEECH_THRESHOLD;
  }

  function chunkDuration(sampleCount) {
    if (!state.captureContext) {
      return 0;
    }
    return (sampleCount / state.captureContext.sampleRate) * 1000;
  }

  function sendActivityStart() {
    if (state.speechActive || !isSocketOpen()) {
      return;
    }
    state.speechActive = true;
    state.silenceMs = 0;
    state.socket.send(JSON.stringify({ type: 'activity_start' }));
  }

  function sendActivityEnd() {
    if (!state.speechActive || !isSocketOpen()) {
      return;
    }
    state.speechActive = false;
    state.silenceMs = 0;
    state.socket.send(JSON.stringify({ type: 'activity_end' }));
  }

  function isSocketOpen() {
    return state.socket && state.socket.readyState === WebSocket.OPEN;
  }

  function clearTranscript() {
    const transcript = document.getElementById('live-transcript');
    if (!transcript) {
      return;
    }
    transcript.innerHTML = '';
    state.transcriptSpeaker = null;
    state.transcriptTextNode = null;
  }

  function appendTranscript(speaker, text) {
    const transcript = document.getElementById('live-transcript');
    if (!transcript || !text) {
      return;
    }
    const entry = document.createElement('p');
    entry.className = 'whitespace-pre-wrap text-sm text-slate-700';
    const label = document.createElement('span');
    label.className = 'font-semibold text-slate-900';
    label.textContent = `${speaker}:`;
    const body = document.createElement('span');
    body.textContent = ` ${text}`;
    entry.appendChild(label);
    entry.appendChild(body);
    transcript.appendChild(entry);
    state.transcriptSpeaker = null;
    state.transcriptTextNode = null;
    transcript.scrollTop = transcript.scrollHeight;
  }

  function appendTranscriptChunk(speaker, text) {
    const transcript = document.getElementById('live-transcript');
    if (!transcript || !text) {
      return;
    }
    if (state.transcriptSpeaker !== speaker || !state.transcriptTextNode) {
      state.transcriptTextNode = createTranscriptEntry(transcript, speaker, text);
      state.transcriptSpeaker = speaker;
      transcript.scrollTop = transcript.scrollHeight;
      return;
    }
    state.transcriptTextNode.textContent += text;
    transcript.scrollTop = transcript.scrollHeight;
  }

  function createTranscriptEntry(transcript, speaker, text) {
    const entry = document.createElement('p');
    entry.className = 'whitespace-pre-wrap text-sm text-slate-700';
    const label = document.createElement('span');
    label.className = 'font-semibold text-slate-900';
    label.textContent = `${speaker}:`;
    const body = document.createElement('span');
    body.textContent = ` ${text}`;
    entry.appendChild(label);
    entry.appendChild(body);
    transcript.appendChild(entry);
    return body;
  }

  function setStatus(stateName, message) {
    const badge = document.getElementById('live-status-badge');
    const text = document.getElementById('live-status-text');
    const startButton = document.getElementById('start-live-button');
    const stopButton = document.getElementById('stop-live-button');
    if (!badge || !text || !startButton || !stopButton) {
      return;
    }
    const config = statusConfig(stateName);
    badge.textContent = config.label;
    badge.className = config.badgeClass;
    text.textContent = message;
    applyButtonState(startButton, config.canStart);
    applyButtonState(stopButton, config.canStop);
  }

  function applyButtonState(button, enabled) {
    button.classList.toggle('opacity-50', !enabled);
    button.classList.toggle('pointer-events-none', !enabled);
    button.classList.toggle('cursor-not-allowed', !enabled);
    button.setAttribute('aria-disabled', enabled ? 'false' : 'true');
  }

  function statusConfig(stateName) {
    if (stateName === 'connecting') {
      return {
        label: 'Verbinde',
        badgeClass: 'rounded-full bg-amber-500 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-white',
        canStart: false,
        canStop: false,
      };
    }
    if (stateName === 'active') {
      return {
        label: 'Live aktiv',
        badgeClass: 'rounded-full bg-emerald-600 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-white',
        canStart: false,
        canStop: true,
      };
    }
    if (stateName === 'stopping') {
      return {
        label: 'Beende',
        badgeClass: 'rounded-full bg-slate-500 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-white',
        canStart: false,
        canStop: false,
      };
    }
    if (stateName === 'error') {
      return {
        label: 'Fehler',
        badgeClass: 'rounded-full bg-red-600 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-white',
        canStart: true,
        canStop: false,
      };
    }
    return {
      label: 'Live aus',
      badgeClass: 'rounded-full bg-slate-500 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-white',
      canStart: true,
      canStop: false,
    };
  }

  function formatError(error) {
    if (!error) {
      return 'Die Live-Funktion konnte nicht gestartet werden.';
    }
    if (error.name === 'NotAllowedError') {
      return 'Bitte erlaube den Mikrofonzugriff fuer das Live-Erlebnis.';
    }
    return error.message || 'Die Live-Funktion konnte nicht gestartet werden.';
  }
  window.projektGrimmLive = {
    startConversation,
    stopConversation,
  };
})();