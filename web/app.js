const $ = id => document.getElementById(id);
const modelNames = {gguf: 'GGUF / F16', gguf_q8: 'GGUF / Q8_0', gguf_q4: 'GGUF / Q4_K_M', mlx: 'MLX / BF16'};
const modelNotes = {gguf: 'Official GGUF · F16 + F16 projector',
  gguf_q8: 'Official GGUF · Q8_0 + Q8_0 projector',
  gguf_q4: 'Official GGUF · Q4_K_M + Q8_0 projector', mlx: 'Self-converted MLX · BF16'};
// Recorded in exports so a result file identifies its weights without consulting the repo.
const modelDetails = {gguf: 'Unquantized. general.file_type=1 (MOSTLY_F16); weights F16×197 + F32×113, audio projector F32×248 + F16×150.',
  gguf_q8: 'general.file_type=7 (MOSTLY_Q8_0); weights Q8_0×197 + F32×113, audio projector F32×248 + Q8_0×147 + F16×3.',
  gguf_q4: 'general.file_type=15 (MOSTLY_Q4_K_M); weights Q4_K×168 + Q6_K×29 + F32×113. The audio projector stays Q8_0; no official Q4 projector exists.',
  mlx: 'Unquantized, converted from the official HF model. All 707 safetensors tensors are BF16; config.json has no quantization field.'};
let selected = 'gguf_q8', active = false, stopping = false, socket = null;
let micStream = null, audioContext = null, worklet = null, flushResolve = null;
let frames = [], lastFrames = [], lastSource = '', lastProcessing = 'file', lastCaptureSettings = null;
let runs = [], current = null;
let replayCancelled = false, levelHistory = Array(56).fill(0);

function notice(message = '') { $('notice').textContent = message; $('notice').hidden = !message; }
function state(text) { $('state').lastChild.textContent = text; }
function lock(value) {
  active = value; document.body.dataset.active = String(value);
  $('settings').disabled = value; $('record').hidden = value; $('stop').hidden = !value;
  $('stop').disabled = false; $('file').disabled = value;
  $('sample').disabled = value;
  document.querySelector('.file-button').classList.toggle('disabled', value);
  $('replay').disabled = value || !lastFrames.length;
  $('caret').hidden = !value; $('t-caret').hidden = !value;
}
function duration(samples) {
  const seconds = Math.floor(samples / 16000);
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}
function metric(id, value) {
  $(id).replaceChildren(document.createTextNode(value == null ? '—' : Math.round(value).toLocaleString()));
  const unit = document.createElement('small'); unit.textContent = ' ms'; $(id).append(unit);
}
// Translating into the spoken language is recognition only; the server skips it too.
function translating() { return $('translate').value !== 'off' && $('translate').value !== $('language').value; }
function showTranslation() {
  const on = translating(), target = $('translate').value;
  document.body.dataset.translate = String(on);
  $('translation-block').hidden = !on; $('lag-metric').hidden = !on;
  $('copy-translation').hidden = !on;
  $('translation-title').textContent = `TRANSLATION → ${target.toUpperCase()}`;
  $('translate-hint').textContent = target === 'off' ? 'Recognition only.'
    : target === $('language').value ? 'Same as the spoken language: recognition only.'
    : `HY-MT1.5 1.8B on ${mtDevice}. Per sentence, grey preview.`;
}
function renderTranslation(text, draft = '') {
  $('t-confirmed').textContent = text; $('t-draft').textContent = draft;
  $('translation-empty').hidden = !!(text || draft); $('translation').hidden = !(text || draft);
  $('copy-translation').disabled = !text;
  const area = $('translation-area');
  if (area.scrollHeight - area.scrollTop - area.clientHeight < 120) area.scrollTop = area.scrollHeight;
}
function renderText(text, draft = '') {
  $('confirmed').textContent = text; $('draft').textContent = draft;
  $('empty').hidden = !!(text || draft); $('transcript').hidden = !(text || draft);
  $('count').textContent = `${[...text].length} chars`;
  $('copy').disabled = !text; $('export').disabled = !text && !runs.length;
  const area = $('transcript-area');
  if (area.scrollHeight - area.scrollTop - area.clientHeight < 160) area.scrollTop = area.scrollHeight;
}
let appVersion = '', mtDevice = 'CPU';
// Service line: the recognition model the next session will use (the one selected
// here, not merely whatever the server has loaded) and the translation model.
async function health() {
  const show = (dot, text) => { $('health-dot').dataset.state = dot; $('health').textContent = text; };
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('Service unavailable');
    const data = await response.json();
    appVersion = data.version; $('version').textContent = `R2D2 // ${data.version}`;
    // An engine whose files are missing, or MLX off a Mac, cannot be chosen.
    for (const button of document.querySelectorAll('.engine')) {
      const missing = data.models[button.dataset.engine] === false;
      button.disabled = missing;
      button.title = missing ? 'Not installed on this machine' : '';
    }
    const mt = data.translation;
    if (mt.device && mtDevice !== mt.device.toUpperCase()) { mtDevice = mt.device.toUpperCase(); showTranslation(); }
    const mtText = !mt.available ? 'MT unavailable' : {ready: 'MT ready', loading: 'MT loading', error: 'MT failed to load', unloaded: 'MT loads on first use'}[mt.state];
    const loaded = modelNames[data.backend];
    if (data.busy) show('busy', `Recognizing · ${loaded} · ${mtText}`);
    else if (data.state === 'loading') show('busy', `Loading ${loaded} · ${mtText}`);
    else if (data.state === 'error' && data.backend === selected) show('error', `${loaded} failed to load · ${mtText}`);
    else if (data.state === 'ready' && data.backend === selected) show(mt.state === 'error' ? 'error' : 'ready', `${loaded} ready · ${mtText}`);
    else show(mt.state === 'error' ? 'error' : 'ready', `${modelNames[selected]} loads on start · ${mtText}`);
  } catch { show('offline', 'Local service not connected'); }
}

for (const button of document.querySelectorAll('.engine')) {
  button.addEventListener('click', async () => {
    if (active) return;
    selected = button.dataset.engine;
    for (const b of document.querySelectorAll('.engine')) {
      b.classList.toggle('selected', b === button); b.setAttribute('aria-pressed', String(b === button));
    }
    $('model-note').textContent = modelNotes[selected];
    $('session-engine').textContent = modelNames[selected];
    // Actual model switch happens on start so browsing controls does not load 4 GB.
    state('Idle'); notice(); health();
  });
}
// Focus mode only restyles the panel in place: the nodes that the stream and
// translation updates write into are never moved or recreated.
function focusMode(on) {
  document.body.dataset.focus = String(on);
  $('focus').setAttribute('aria-pressed', String(on));
  $('focus').textContent = on ? '⤡ Restore' : '⤢ Expand';
  for (const id of ['transcript-area', 'translation-area']) $(id).scrollTop = $(id).scrollHeight;
}
$('focus').addEventListener('click', () => focusMode(document.body.dataset.focus !== 'true'));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && document.body.dataset.focus === 'true') focusMode(false);
});
$('translate').addEventListener('change', showTranslation);
$('language').addEventListener('change', showTranslation);
$('processing').addEventListener('change', () => {
  $('capture-hint').textContent = $('processing').value === 'raw'
    ? 'Denoise, echo cancel and AGC all off.'
    : 'Denoise and echo cancel on; AGC off.';
});
async function devices() {
  if (!navigator.mediaDevices) return;
  const previous = $('microphone').value;
  const inputs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput');
  $('microphone').replaceChildren(new Option('Default input', ''));
  for (const [i, input] of inputs.entries()) {
    if (input.deviceId && input.deviceId !== 'default') $('microphone').add(new Option(input.label || `Microphone ${i + 1}`, input.deviceId));
  }
  if ([...$('microphone').options].some(o => o.value === previous)) $('microphone').value = previous;
}

function drawMeter() {
  const canvas = $('meter'), ctx = canvas.getContext('2d'), scale = devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  canvas.width = width * scale; canvas.height = height * scale; ctx.scale(scale, scale);
  ctx.clearRect(0, 0, width, height);
  const spacing = width / levelHistory.length;
  for (let i = 0; i < levelHistory.length; i++) {
    const h = Math.max(1, levelHistory[i] * height * .9);
    ctx.fillStyle = levelHistory[i] > .01 ? '#b0b0b0' : '#3a3a3a';
    ctx.fillRect(i * spacing, (height - h) / 2, Math.max(1, spacing - 2), h);
  }
}
function showLevel(frame) {
  let squares = 0;
  for (const sample of frame) squares += (sample / 32768) ** 2;
  const rms = Math.sqrt(squares / frame.length), db = rms > 0 ? 20 * Math.log10(rms) : -96;
  $('db').textContent = `${Math.max(-96, db).toFixed(0)} dBFS`;
  levelHistory.push(Math.min(1, rms * 5)); levelHistory.shift(); drawMeter();
}
function sendFrame(buffer) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  if (socket.bufferedAmount > 320000) { fail('Network send backlog passed 10 s; recording stopped.'); return; }
  const frame = new Int16Array(buffer);
  frames.push(frame.slice());
  current.samples += frame.length;
  $('timer').textContent = duration(current.samples);
  showLevel(frame); socket.send(buffer);
  if (current.samples >= 16000 * 300 && !stopping) stop();
}

function openSocket() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/stream`);
    socket = ws;
    let ready = false, finished = false;
    const timeout = setTimeout(() => { reject(new Error('Model load timed out')); ws.close(); }, 150000);
    ws.onopen = () => ws.send(JSON.stringify({backend: selected, language: $('language').value, context: $('context').value,
      translate: translating() ? $('translate').value : 'off'}));
    ws.onmessage = event => {
      let data;
      try { data = JSON.parse(event.data); } catch { fail('The service returned an invalid message'); return; }
      if (data.type === 'loading') state('Loading model…');
      if (data.type === 'ready') {
        clearTimeout(timeout); ready = true; state('Listening');
        current.translation = data.translate ? {target: data.target, text: '', updates: [], sentences: []} : null;
        $('translation-state').textContent = data.translate ? `HY-MT1.5 · 1.8B · ${mtDevice}` : current.translate_error ? 'Translation unavailable' : 'Not translating';
        resolve();
      }
      if (data.type === 'translation_error') {
        current.translate_error = data.message; $('translation-state').textContent = 'Translation unavailable'; notice(data.message);
      }
      if (data.type === 'translation' && current.translation) {
        const {type, sentences, ...update} = data;
        current.translation.text = data.text;
        current.translation.updates.push({...update, received_ms: Math.round(performance.now() - current.started)});
        if (sentences) current.translation.sentences = sentences;
        if (data.error) current.translate_error = data.error;
        renderTranslation(data.text, data.draft);
        if (data.lag_ms != null) metric('lag', data.lag_ms);
      }
      if (data.type === 'transcript') {
        current.text = data.text; current.language = data.language;
        current.updates.push({...data, received_ms: Math.round(performance.now() - current.started)});
        if (data.reset) current.resets.push({step: data.step, audio_ms: data.audio_ms, reason: data.reset});
        renderText(data.text, data.draft);
        metric('first', data.first_text_ms); metric('decode', data.decode_ms); metric('backlog', data.backlog_ms);
        $('detected').textContent = data.language || 'Detecting language';
        if (data.backlog_ms > 1000) notice(`Recognition is ${(data.backlog_ms / 1000).toFixed(1)} s behind the input and merging steps to catch up${data.hops > 1 ? ` (this step merged ${data.hops} chunks)` : ''}. Remaining audio is processed after you stop.`);
        else if (!stopping && !current.translate_error) notice();
      }
      if (data.type === 'done') { finished = true; complete(); }
      if (data.type === 'error') {
        clearTimeout(timeout); finished = true;
        if (!ready) reject(new Error(data.message)); else fail(data.message);
      }
    };
    ws.onerror = () => { if (!ready) { clearTimeout(timeout); reject(new Error('Cannot reach the local recognition service')); } };
    ws.onclose = () => {
      clearTimeout(timeout);
      if (!ready) reject(new Error('The recognition service disconnected before it was ready'));
      else if (!finished && active) fail('Recognition connection lost; the text and audio received so far can still be exported or replayed.');
    };
  });
}
function begin(source, processing) {
  stopping = false; replayCancelled = false; frames = []; notice(); lock(true);
  current = {id: crypto.randomUUID(), backend: selected, source, processing,
    model: modelNotes[selected], model_detail: modelDetails[selected], language: $('language').value, context: $('context').value, started: performance.now(),
    date: new Date().toISOString(), samples: 0, text: '', updates: [], resets: [], note: '',
    translation: null, translate_error: ''};
  renderText(''); renderTranslation(''); showTranslation();
  $('translation-state').textContent = translating() ? `HY-MT1.5 · 1.8B · ${mtDevice}` : 'Not translating'; $('timer').textContent = '00:00';
  $('session-engine').textContent = modelNames[selected];
  $('source-label').textContent = source === 'microphone' ? 'MICROPHONE' : 'AUDIO REPLAY';
  ['first', 'decode', 'backlog', 'lag'].forEach(id => metric(id, null));
  state('Preparing input…');
}
async function releaseMic() {
  micStream?.getTracks().forEach(track => track.stop()); micStream = null;
  if (audioContext && audioContext.state !== 'closed') await audioContext.close();
  audioContext = null; worklet = null;
}
async function record() {
  if (active) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    notice('The microphone needs a secure page. Open http://localhost:8765 on this machine, or serve HTTPS for LAN access.'); return;
  }
  begin('microphone', $('processing').value);
  $('stop').disabled = true;
  try {
    // Create/resume audio in the click gesture, before a potentially long model load.
    audioContext = new AudioContext({sampleRate: 16000});
    await audioContext.resume();
    const processed = $('processing').value === 'browser';
    micStream = await navigator.mediaDevices.getUserMedia({audio: {
      deviceId: $('microphone').value ? {exact: $('microphone').value} : undefined,
      channelCount: 1, sampleRate: 16000, noiseSuppression: processed,
      echoCancellation: processed, autoGainControl: false,
    }});
    current.captureSettings = micStream.getAudioTracks()[0].getSettings();
    await devices();
    await openSocket();
    if (!active) return;
    current.started = performance.now();
    await audioContext.audioWorklet.addModule('/static/capture-worklet.js');
    worklet = new AudioWorkletNode(audioContext, 'pcm-capture');
    worklet.port.onmessage = ({data}) => {
      if (data === 'flushed') { flushResolve?.(); flushResolve = null; }
      else sendFrame(data);
    };
    const input = audioContext.createMediaStreamSource(micStream);
    const filter = audioContext.createBiquadFilter(); filter.type = 'lowpass'; filter.frequency.value = 7200;
    const mute = audioContext.createGain(); mute.gain.value = 0;
    if (audioContext.sampleRate > 16000 * 1.001) input.connect(filter).connect(worklet);
    else input.connect(worklet);
    worklet.connect(mute).connect(audioContext.destination);
    $('stop').disabled = false;
  } catch (error) {
    const labels = {NotAllowedError: 'Microphone permission denied. Allow it from the browser address bar and try again.',
      NotFoundError: 'No microphone found.', NotReadableError: 'The microphone cannot be opened; another app may be using it.'};
    fail(labels[error.name] || error.message);
  }
}
async function stop() {
  if (!active || stopping) return;
  stopping = true; replayCancelled = true; $('stop').disabled = true; state('Finishing…');
  if (worklet) {
    await Promise.race([new Promise(resolve => { flushResolve = resolve; worklet.port.postMessage('stop'); }),
      new Promise(resolve => setTimeout(resolve, 1000))]);
  }
  await releaseMic();
  if (socket?.readyState === WebSocket.OPEN) socket.send('stop');
}
function keepAudio() {
  if (frames.length) {
    lastFrames = frames.map(f => f.slice()); lastSource = current?.source || '';
    lastProcessing = current?.processing || 'file'; lastCaptureSettings = current?.captureSettings || null;
  }
}
async function fail(message) {
  replayCancelled = true; stopping = true;
  await releaseMic();
  keepAudio();
  lock(false); socket?.close(); socket = null;
  state('Stopped'); notice(message);
  if (current?.text) saveRun('interrupted');
  health();
}
async function complete() {
  await releaseMic();
  // Include a digest in exported runs so A/B input identity can be verified.
  if (window.isSecureContext && frames.length) {
    const pcm = new Int16Array(current.samples); let offset = 0;
    for (const frame of frames) { pcm.set(frame, offset); offset += frame.length; }
    const digest = await crypto.subtle.digest('SHA-256', pcm.buffer);
    current.audio_sha256 = [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('');
  }
  keepAudio(); lock(false); stopping = false;
  state('Complete'); notice(); saveRun('complete'); health();
}
function saveRun(result) {
  if (!current || runs.some(run => run.id === current.id)) return;
  current.result = result; runs.unshift(current);
  $('runs-section').hidden = false;
  const row = document.createElement('div'); row.className = 'run';
  const meta = document.createElement('div'); meta.className = 'run-meta';
  const title = document.createElement('b'); title.textContent = modelNames[current.backend];
  const info = document.createElement('div'); info.textContent = `${duration(current.samples)} · ${current.processing === 'browser' ? 'Browser processing' : current.processing === 'raw' ? 'Raw input' : 'Imported audio'}`;
  meta.append(title, info);
  if (current.resets.length) {
    const resets = document.createElement('div');
    resets.textContent = `Decoder rebuilt ${current.resets.length}×`;
    resets.title = current.resets.map(r => `${(r.audio_ms / 1000).toFixed(1)}s ${r.reason}`).join('\n');
    meta.append(resets);
  }
  const body = document.createElement('div'), text = document.createElement('p');
  text.textContent = current.text || '(no text recognized)';
  const note = document.createElement('textarea'); note.rows = 1;
  note.placeholder = 'Listening notes: fluency, missed words, noise misfires…'; note.setAttribute('aria-label', `${modelNames[current.backend]} listening notes`);
  const saved = current; note.addEventListener('input', () => saved.note = note.value);
  body.append(text);
  if (current.translation?.text) {
    const translated = document.createElement('p'); translated.className = 'run-translation';
    translated.textContent = current.translation.text; body.append(translated);
  }
  body.append(note); row.append(meta, body); $('runs').prepend(row); $('export').disabled = false;
}
async function replay(inputFrames, source, processing, captureSettings = null) {
  if (active) return;
  const snapshot = inputFrames.map(f => f.slice());
  begin(source, processing); $('stop').disabled = true;
  if (captureSettings) current.captureSettings = captureSettings;
  try {
    await openSocket();
    $('stop').disabled = false; state('Replaying');
    const start = performance.now(); let sent = 0;
    current.started = start;
    for (const frame of snapshot) {
      const due = start + (sent + frame.length) / 16;
      await new Promise(resolve => setTimeout(resolve, Math.max(0, due - performance.now())));
      if (replayCancelled || !active) return;
      sendFrame(frame.slice().buffer); sent += frame.length;
    }
    await stop();
  } catch (error) { fail(error.message); }
}
$('record').addEventListener('click', record);
$('stop').addEventListener('click', stop);
$('replay').addEventListener('click', () => {
  replay(lastFrames, lastSource.startsWith('replay:') ? lastSource : `replay:${lastSource}`, lastProcessing, lastCaptureSettings);
});
async function importAudio(bytes, name) {
  if (active) return;
  let decoder;
  try {
    decoder = new AudioContext();
    const decoded = await decoder.decodeAudioData(bytes);
    if (decoded.duration > 300) throw new Error('a single clip is limited to 5 minutes');
    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
    const source = offline.createBufferSource(); source.buffer = decoded;
    source.connect(offline.destination); source.start();
    const mono = (await offline.startRendering()).getChannelData(0);
    const packets = [];
    for (let i = 0; i < mono.length; i += 2560) {
      packets.push(Int16Array.from(mono.subarray(i, i + 2560), sample => Math.round(Math.max(-1, Math.min(32767 / 32768, sample)) * 32768)));
    }
    await decoder.close(); decoder = null;
    await replay(packets, name, 'file');
  } catch (error) { notice(`Cannot import audio: ${error.message}`); }
  finally { if (decoder) await decoder.close(); }
}
$('file').addEventListener('change', async event => {
  const file = event.target.files[0]; event.target.value = '';
  if (!file || active) return;
  if (file.size > 100 * 1024 * 1024) { notice('Choose audio under 100 MB and no longer than 5 minutes.'); return; }
  await importAudio(await file.arrayBuffer(), file.name);
});
$('sample').addEventListener('click', async () => {
  if (active) return;
  try {
    const response = await fetch('/api/sample');
    if (!response.ok) throw new Error('Sample audio unavailable');
    await importAudio(await response.arrayBuffer(), 'official-test.wav');
  } catch (error) { notice(error.message); }
});
$('copy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('confirmed').textContent); $('copy').textContent = 'Copied'; setTimeout(() => $('copy').textContent = 'Copy text', 1500); }
  catch { notice('The browser blocked copying; select the text and copy it manually.'); }
});
$('copy-translation').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('t-confirmed').textContent); $('copy-translation').textContent = 'Copied'; setTimeout(() => $('copy-translation').textContent = 'Copy translation', 1500); }
  catch { notice('The browser blocked copying; select the text and copy it manually.'); }
});
$('export').addEventListener('click', () => {
  const entries = current && !runs.some(r => r.id === current.id) ? [current, ...runs] : runs;
  const blob = new Blob([JSON.stringify({app: 'R2D2', version: appVersion, policy: '160ms hop / 160ms lookahead / 8s window / 1 token rollback / merges up to 3 hops when behind',
    translation_policy: `HY-MT1.5-1.8B Q4_K_M on ${mtDevice} / greedy / settle per closed sentence / latest-wins draft`, runs: entries}, null, 2)], {type: 'application/json'});
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob);
  link.download = `r2d2-${new Date().toISOString().replaceAll(':', '-')}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});
window.addEventListener('beforeunload', () => { micStream?.getTracks().forEach(t => t.stop()); socket?.close(); });
window.addEventListener('resize', drawMeter);
navigator.mediaDevices?.addEventListener('devicechange', () => devices().catch(() => {}));
showTranslation(); drawMeter(); health(); devices().catch(() => {}); setInterval(health, 5000);
