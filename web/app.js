const $ = id => document.getElementById(id);
const modelNames = {gguf: 'GGUF / F16', gguf_q8: 'GGUF / Q8_0', gguf_q4: 'GGUF / Q4_K_M', mlx: 'MLX / BF16'};
const modelNotes = {gguf: '官方 GGUF · F16 主权重 + F16 音频投影',
  gguf_q8: '官方 GGUF · Q8_0 主权重 + Q8_0 音频投影',
  gguf_q4: '官方 GGUF · Q4_K_M 主权重 + Q8_0 音频投影', mlx: '自行转换 MLX · BF16'};
// Recorded in exports so a result file identifies its weights without consulting the repo.
const modelDetails = {gguf: '未量化。general.file_type=1 (MOSTLY_F16)；主权重 F16×197 + F32×113，音频投影 F32×248 + F16×150。',
  gguf_q8: 'general.file_type=7 (MOSTLY_Q8_0)；主权重 Q8_0×197 + F32×113，音频投影 F32×248 + Q8_0×147 + F16×3。',
  gguf_q4: 'general.file_type=15 (MOSTLY_Q4_K_M)；主权重 Q4_K×168 + Q6_K×29 + F32×113。音频投影沿用 Q8_0，官方未发布 Q4 投影。',
  mlx: '未量化，自官方 HF 模型转换。safetensors 707 个张量全部 BF16，config.json 无 quantization 字段。'};
let selected = 'gguf', active = false, stopping = false, socket = null;
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
  $('caret').hidden = !value;
}
function duration(samples) {
  const seconds = Math.floor(samples / 16000);
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}
function metric(id, value) {
  $(id).replaceChildren(document.createTextNode(value == null ? '—' : Math.round(value).toLocaleString()));
  const unit = document.createElement('small'); unit.textContent = ' ms'; $(id).append(unit);
}
function renderText(text, draft = '') {
  $('confirmed').textContent = text; $('draft').textContent = draft;
  $('empty').hidden = !!(text || draft); $('transcript').hidden = !(text || draft);
  $('count').textContent = `${[...text].length} 字符`;
  $('copy').disabled = !text; $('export').disabled = !text && !runs.length;
  const area = $('transcript-area');
  if (area.scrollHeight - area.scrollTop - area.clientHeight < 160) area.scrollTop = area.scrollHeight;
}
async function health() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('服务不可用');
    const data = await response.json();
    const labels = {ready: '模型已就绪', loading: '模型加载中', unloaded: '按下开始时加载模型', error: '模型加载失败'};
    $('health').textContent = `${data.busy ? '识别中' : labels[data.state]} · ${modelNames[data.backend]}`;
  } catch { $('health').textContent = '本机服务未连接'; }
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
    state('待机'); notice();
  });
}
$('processing').addEventListener('change', () => {
  $('capture-hint').textContent = $('processing').value === 'raw'
    ? '浏览器降噪、回声消除、自动增益均关闭。'
    : '降噪、回声消除开启；自动增益关闭。';
});
async function devices() {
  if (!navigator.mediaDevices) return;
  const previous = $('microphone').value;
  const inputs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput');
  $('microphone').replaceChildren(new Option('系统默认输入', ''));
  for (const [i, input] of inputs.entries()) {
    if (input.deviceId && input.deviceId !== 'default') $('microphone').add(new Option(input.label || `麦克风 ${i + 1}`, input.deviceId));
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
  if (socket.bufferedAmount > 320000) { fail('网络发送积压超过 10 秒，录音已停止。'); return; }
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
    const timeout = setTimeout(() => { reject(new Error('模型加载超时')); ws.close(); }, 150000);
    ws.onopen = () => ws.send(JSON.stringify({backend: selected, language: $('language').value, context: $('context').value}));
    ws.onmessage = event => {
      let data;
      try { data = JSON.parse(event.data); } catch { fail('服务返回了无效消息'); return; }
      if (data.type === 'loading') state('加载模型中…');
      if (data.type === 'ready') { clearTimeout(timeout); ready = true; state('正在聆听'); resolve(); }
      if (data.type === 'transcript') {
        current.text = data.text; current.language = data.language;
        current.updates.push({...data, received_ms: Math.round(performance.now() - current.started)});
        if (data.reset) current.resets.push({step: data.step, audio_ms: data.audio_ms, reason: data.reset});
        renderText(data.text, data.draft);
        metric('first', data.first_text_ms); metric('decode', data.decode_ms); metric('backlog', data.backlog_ms);
        $('detected').textContent = data.language || '识别语言中';
        if (data.backlog_ms > 1000) notice(`识别落后输入 ${(data.backlog_ms / 1000).toFixed(1)} 秒，正在合并步长追赶${data.hops > 1 ? `（本步合并 ${data.hops} 块）` : ''}。停止录音后会继续处理余下音频。`);
        else if (!stopping) notice();
      }
      if (data.type === 'done') { finished = true; complete(); }
      if (data.type === 'error') {
        clearTimeout(timeout); finished = true;
        if (!ready) reject(new Error(data.message)); else fail(data.message);
      }
    };
    ws.onerror = () => { if (!ready) { clearTimeout(timeout); reject(new Error('无法连接本机识别服务')); } };
    ws.onclose = () => {
      clearTimeout(timeout);
      if (!ready) reject(new Error('识别服务在就绪前断开连接'));
      else if (!finished && active) fail('识别连接已断开；已收到的文字和音频仍可导出或重放。');
    };
  });
}
function begin(source, processing) {
  stopping = false; replayCancelled = false; frames = []; notice(); lock(true);
  current = {id: crypto.randomUUID(), backend: selected, source, processing,
    model: modelNotes[selected], model_detail: modelDetails[selected], language: $('language').value, context: $('context').value, started: performance.now(),
    date: new Date().toISOString(), samples: 0, text: '', updates: [], resets: [], note: ''};
  renderText(''); $('timer').textContent = '00:00';
  $('session-engine').textContent = modelNames[selected];
  $('source-label').textContent = source === 'microphone' ? 'MICROPHONE' : 'AUDIO REPLAY';
  ['first', 'decode', 'backlog'].forEach(id => metric(id, null));
  state('准备输入…');
}
async function releaseMic() {
  micStream?.getTracks().forEach(track => track.stop()); micStream = null;
  if (audioContext && audioContext.state !== 'closed') await audioContext.close();
  audioContext = null; worklet = null;
}
async function record() {
  if (active) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    notice('麦克风需要安全页面。请在本机打开 http://localhost:8765，或为局域网访问配置 HTTPS。'); return;
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
    const labels = {NotAllowedError: '麦克风权限未获允许。请在浏览器地址栏允许麦克风，然后重试。',
      NotFoundError: '没有找到可用麦克风。', NotReadableError: '麦克风无法打开，可能正在被其他应用占用。'};
    fail(labels[error.name] || error.message);
  }
}
async function stop() {
  if (!active || stopping) return;
  stopping = true; replayCancelled = true; $('stop').disabled = true; state('正在完成…');
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
  state('已停止'); notice(message);
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
  state('识别完成'); notice(); saveRun('complete'); health();
}
function saveRun(result) {
  if (!current || runs.some(run => run.id === current.id)) return;
  current.result = result; runs.unshift(current);
  $('runs-section').hidden = false;
  const row = document.createElement('div'); row.className = 'run';
  const meta = document.createElement('div'); meta.className = 'run-meta';
  const title = document.createElement('b'); title.textContent = modelNames[current.backend];
  const info = document.createElement('div'); info.textContent = `${duration(current.samples)} · ${current.processing === 'browser' ? '浏览器降噪' : current.processing === 'raw' ? '原始输入' : '导入音频'}`;
  meta.append(title, info);
  if (current.resets.length) {
    const resets = document.createElement('div');
    resets.textContent = `解码器重置 ${current.resets.length} 次`;
    resets.title = current.resets.map(r => `${(r.audio_ms / 1000).toFixed(1)}s ${r.reason}`).join('\n');
    meta.append(resets);
  }
  const body = document.createElement('div'), text = document.createElement('p');
  text.textContent = current.text || '（没有识别到文字）';
  const note = document.createElement('textarea'); note.rows = 1;
  note.placeholder = '听感记录：流畅度、漏字、噪声误识别…'; note.setAttribute('aria-label', `${modelNames[current.backend]} 听感记录`);
  const saved = current; note.addEventListener('input', () => saved.note = note.value);
  body.append(text, note); row.append(meta, body); $('runs').prepend(row); $('export').disabled = false;
}
async function replay(inputFrames, source, processing, captureSettings = null) {
  if (active) return;
  const snapshot = inputFrames.map(f => f.slice());
  begin(source, processing); $('stop').disabled = true;
  if (captureSettings) current.captureSettings = captureSettings;
  try {
    await openSocket();
    $('stop').disabled = false; state('实时重放中');
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
    if (decoded.duration > 300) throw new Error('单段音频上限为 5 分钟');
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
  } catch (error) { notice(`音频无法导入：${error.message}`); }
  finally { if (decoder) await decoder.close(); }
}
$('file').addEventListener('change', async event => {
  const file = event.target.files[0]; event.target.value = '';
  if (!file || active) return;
  if (file.size > 100 * 1024 * 1024) { notice('请选择小于 100 MB、时长不超过 5 分钟的音频。'); return; }
  await importAudio(await file.arrayBuffer(), file.name);
});
$('sample').addEventListener('click', async () => {
  if (active) return;
  try {
    const response = await fetch('/api/sample');
    if (!response.ok) throw new Error('示例音频不可用');
    await importAudio(await response.arrayBuffer(), 'official-test.wav');
  } catch (error) { notice(error.message); }
});
$('copy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('confirmed').textContent); $('copy').textContent = '已复制'; setTimeout(() => $('copy').textContent = '复制文字', 1500); }
  catch { notice('浏览器未允许复制，请选择文字后手动复制。'); }
});
$('export').addEventListener('click', () => {
  const entries = current && !runs.some(r => r.id === current.id) ? [current, ...runs] : runs;
  const blob = new Blob([JSON.stringify({app: 'R2D2', version: '0.1', policy: '160ms hop / 160ms lookahead / 8s window / 1 token rollback / merges up to 3 hops when behind', runs: entries}, null, 2)], {type: 'application/json'});
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob);
  link.download = `r2d2-${new Date().toISOString().replaceAll(':', '-')}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});
window.addEventListener('beforeunload', () => { micStream?.getTracks().forEach(t => t.stop()); socket?.close(); });
window.addEventListener('resize', drawMeter);
navigator.mediaDevices?.addEventListener('devicechange', () => devices().catch(() => {}));
drawMeter(); health(); devices().catch(() => {}); setInterval(health, 5000);
