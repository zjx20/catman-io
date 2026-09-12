// 对话 demo 页：麦克风 → AudioWorklet(16 kHz int16) → WebSocket → 服务端整条管线；
// 服务端回来的二进制帧是要播的 PCM（Web Audio 排队播放，收到 audio_stop 就清掉），文本帧是状态 / 仪表 / 回合结果。
(() => {
  const $ = (id) => document.getElementById(id);
  const STATE_TEXT = { idle: '待唤醒', listening: '聆听中，请说话', thinking: '思考中', speaking: '播报中',
                       followup: '跟进：可以直接接着说' };
  const STATUS_TEXT = { ok: '完成', no_speech: '没听到说话', empty: '没听清', cancelled: '被打断',
                        timeout: '超时', error: '出错' };
  const state = { ws: null, ctx: null, stream: null, node: null, running: false,
                  playCtx: null, gain: null, nextTime: 0, sources: [], volume: 0.8,
                  hello: null, curState: 'idle', stateSince: 0, turns: 0 };
  window.dialogDemo = state; // 方便自动化测试读取

  // ---------------------------------------------------------------- WebSocket
  function connect() {
    return new Promise((resolve, reject) => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${proto}://${location.host}/ws/dialog`);
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => { state.ws = ws; setStatus('已连接，正在组装管线（加载识别模型要几秒）…'); resolve(ws); };
      ws.onerror = () => reject(new Error('WebSocket 连接失败'));
      ws.onclose = () => { state.ws = null; setStatus('服务端连接已断开'); stop(); };
      ws.onmessage = (ev) => {
        if (ev.data instanceof ArrayBuffer) playPcm(ev.data);
        else onMessage(JSON.parse(ev.data));
      };
    });
  }
  function send(obj) { if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify(obj)); }

  function onMessage(m) {
    if (m.type === 'hello') {
      state.hello = m;
      renderParts(m);
      $('thrLine').style.left = `${m.threshold * 100}%`;
      setStatus(m.asr ? `就绪，说「小貓人」（或按「按钮唤醒」）` : '识别器未就绪：只能看到唤醒和切句，说话不会被识别');
      $('wake').disabled = false; $('stopSpeak').disabled = false;
    } else if (m.type === 'meter') {
      $('wakeV').textContent = m.wake.toFixed(2);
      $('wakeFill').style.width = `${Math.min(100, m.wake * 100)}%`;
      $('wakeFill').classList.toggle('hit', state.hello && m.wake >= state.hello.threshold);
      $('vadV').textContent = m.vad.toFixed(2);
      $('vadFill').style.width = `${Math.min(100, m.vad * 100)}%`;
      $('level').textContent = `${m.level_db.toFixed(0)} dBFS`;
      $('dropped').textContent = m.dropped;
      if (m.state !== state.curState) setState(m.state);
    } else if (m.type === 'state') {
      setState(m.state);
    } else if (m.type === 'turn') {
      addTurn(m);
    } else if (m.type === 'audio_stop') {
      flushPlayback();
    } else if (m.type === 'error') {
      setStatus(`服务端错误：${m.message}`);
    }
  }

  function setState(s) {
    state.curState = s; state.stateSince = performance.now();
    document.querySelectorAll('#states .st').forEach((el) => el.classList.toggle('on', el.dataset.s === s));
    $('stateText').textContent = STATE_TEXT[s] || s;
  }

  function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

  function addTurn(m) {
    state.turns++;
    const lat = m.latency || {};
    const parts = [];
    if (lat.speech_start != null) parts.push(`唤醒后 ${lat.speech_start} s 开口`);
    if (lat.speech_end != null) parts.push(`${lat.speech_end} s 说完`);
    if (lat.asr != null) parts.push(`识别 ${lat.asr} s`);
    if (lat.first_audio != null) parts.push(`说完到出声 ${lat.first_audio} s`);
    if (lat.reply_done != null) parts.push(`全程 ${lat.reply_done} s`);
    const notes = [new Date().toLocaleTimeString()];
    if (m.followup) notes.push('跟进');
    if (m.barge_in) notes.push('打断了上一句');
    if (m.reprompted) notes.push('重听了一次');
    if (m.wake_model === 'manual') notes.push('按钮唤醒');
    else if (m.wake_score != null) notes.push(`唤醒分 ${m.wake_score.toFixed(2)}`);
    const div = document.createElement('div'); div.className = 'turn';
    div.innerHTML =
      `<div><span class="tag ${esc(m.status)}">${STATUS_TEXT[m.status] || esc(m.status)}</span> ` +
      `<span class="hint">${notes.map(esc).join(' · ')}</span></div>` +
      (m.text ? `<div><span class="who">你</span>${esc(m.text)}` +
                (m.intent ? `<span class="intent">${esc(m.intent)}</span>` : '') + '</div>' : '') +
      (m.reply ? `<div><span class="who">貓</span>${esc(m.reply)}</div>` : '') +
      (m.error ? `<div class="err">${esc(m.error)}</div>` : '') +
      (parts.length ? `<div class="hint">${parts.join(' · ')}</div>` : '');
    $('turns').prepend(div);
    $('turnsHint').textContent = `${state.turns} 回合`;
  }

  function renderParts(h) {
    const on = (v, text) => v ? text : `<span class="off">关</span>`;
    const items = [
      ['唤醒模型', esc(h.models.join(', ')) + `（阈值 ${h.threshold}）`],
      ['识别', h.asr ? esc(h.asr_backend) : `<span class="off">未就绪</span>`],
      ['合成', h.tts ? esc(h.tts_voice) : `<span class="off">关（回复只显示文字）</span>`],
      ['意图规则', `${h.intents} 条`],
      ['LLM 意图层', on(h.llm, '开')],
      ['对话模型', on(h.brain, '开')],
      ['catman', on(h.catman, '已接')],
      ['提示音', on(h.earcons, '开')],
      ['跟进窗口', `${h.followup_seconds} s`],
      ['说完判定', `停顿 ${h.trailing_silence_ms} ms`],
    ];
    $('parts').innerHTML = items.map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
    const hints = [];
    if (!h.asr) hints.push('识别器没加载：在服务端 pip install "catman-io[asr]" 并 catman-io setup --asr，再重启 webdemo。');
    if (!h.tts) hints.push('合成关着：config 里 tts.backend 设成 edge（要联网）就能听到回复。');
    if (!h.brain) hints.push('没接对话模型：规则没命中的话会回「唔識答」；config 的 brain.llm 填上就能闲聊。');
    $('partsHint').textContent = hints.join(' ');
  }

  // ---------------------------------------------------------------- 播放（Web Audio 排队）
  function ensurePlayer() {
    if (!state.playCtx) {
      state.playCtx = new AudioContext();
      state.gain = state.playCtx.createGain();
      state.gain.gain.value = state.volume;
      state.gain.connect(state.playCtx.destination);
    }
    if (state.playCtx.state === 'suspended') state.playCtx.resume();
  }
  function playPcm(buf) {
    ensurePlayer();
    const ctx = state.playCtx;
    const i16 = new Int16Array(buf);
    if (!i16.length) return;
    const ab = ctx.createBuffer(1, i16.length, 16000);
    const ch = ab.getChannelData(0);
    for (let i = 0; i < i16.length; i++) ch[i] = i16[i] / 32768;
    const src = ctx.createBufferSource();
    src.buffer = ab; src.connect(state.gain);
    const startAt = Math.max(state.nextTime, ctx.currentTime + 0.03);
    src.start(startAt);
    state.nextTime = startAt + ab.duration;
    state.sources.push(src);
    src.onended = () => { state.sources = state.sources.filter((s) => s !== src); };
  }
  function flushPlayback() {
    for (const s of state.sources) { try { s.stop(); } catch (e) {} }
    state.sources = []; state.nextTime = 0;
  }
  $('volume').addEventListener('input', () => {
    state.volume = Number($('volume').value);
    if (state.gain) state.gain.gain.value = state.volume;
  });

  // ---------------------------------------------------------------- 麦克风
  async function start() {
    try {
      $('start').disabled = true;
      ensurePlayer(); // 在用户点击里创建，浏览器才允许出声
      if (!state.ws) await connect();
      const constraints = { audio: {
        channelCount: 1,
        echoCancellation: $('ec').checked, noiseSuppression: $('ns').checked, autoGainControl: $('agc').checked,
      } };
      const dev = $('device').value;
      if (dev) constraints.audio.deviceId = { exact: dev };
      state.stream = await navigator.mediaDevices.getUserMedia(constraints);
      await fillDevices();
      let ctx;
      try { ctx = new AudioContext({ sampleRate: 16000 }); } catch (e) { ctx = new AudioContext(); }
      state.ctx = ctx;
      await ctx.resume();
      const src = ctx.createMediaStreamSource(state.stream);
      await ctx.audioWorklet.addModule('/static/pcm-worklet.js');
      const node = new AudioWorkletNode(ctx, 'pcm-capture', { processorOptions: { targetRate: 16000 } });
      node.port.onmessage = (e) => { if (state.ws && state.ws.readyState === 1) state.ws.send(e.data); };
      src.connect(node);
      state.node = node;
      state.running = true;
      $('stop').disabled = false;
    } catch (e) {
      setStatus(`无法开始：${e.message || e}`);
      $('start').disabled = false;
    }
  }

  function stop() {
    state.running = false;
    if (state.node) { try { state.node.disconnect(); } catch (e) {} state.node = null; }
    if (state.stream) { state.stream.getTracks().forEach((t) => t.stop()); state.stream = null; }
    if (state.ctx) { state.ctx.close(); state.ctx = null; }
    if (state.ws) { state.ws.close(); state.ws = null; }
    flushPlayback();
    $('start').disabled = false; $('stop').disabled = true; $('wake').disabled = true; $('stopSpeak').disabled = true;
    setState('idle');
  }

  async function fillDevices() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const sel = $('device'); const cur = sel.value;
    sel.innerHTML = '<option value="">默认</option>';
    for (const d of devices.filter((d) => d.kind === 'audioinput')) {
      const o = document.createElement('option');
      o.value = d.deviceId; o.textContent = d.label || `麦克风 ${sel.length}`;
      sel.appendChild(o);
    }
    sel.value = cur;
  }

  // ---------------------------------------------------------------- 控件
  $('start').onclick = start;
  $('stop').onclick = stop;
  $('wake').onclick = () => send({ type: 'wake' });
  $('stopSpeak').onclick = () => send({ type: 'stop' });
  function setStatus(s) { $('status').textContent = s; }
  setInterval(() => {
    if (!state.running) return;
    const sec = (performance.now() - state.stateSince) / 1000;
    $('stateSince').textContent = `已持续 ${sec.toFixed(0)} s`;
  }, 500);
  if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) fillDevices().catch(() => {});
})();
