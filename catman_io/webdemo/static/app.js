// 页面逻辑：麦克风 → AudioWorklet(16 kHz int16) → WebSocket → 服务端检测 → 逐帧分数画图、命中记录、保存样本。
(() => {
  const $ = (id) => document.getElementById(id);
  const WINDOW_SECONDS = 20;
  const FRAME_SECONDS = 0.08;

  const state = { ws: null, ctx: null, stream: null, node: null, running: false,
                  hits: 0, frames: 0, startedAt: 0, lastScore: 0, threshold: 0.5,
                  points: [], hitPoints: [], pendingAutoSave: null };
  window.demo = state; // 方便自动化测试读取

  // ---------------------------------------------------------------- WebSocket
  function connect() {
    return new Promise((resolve, reject) => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => { state.ws = ws; setStatus('已连接服务端'); resolve(ws); };
      ws.onerror = (e) => reject(new Error('WebSocket 连接失败'));
      ws.onclose = () => { state.ws = null; setStatus('服务端连接已断开'); stop(); };
      ws.onmessage = (ev) => onMessage(JSON.parse(ev.data));
    });
  }

  function send(obj) { if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify(obj)); }

  function onMessage(m) {
    if (m.type === 'hello') {
      $('threshold').value = m.threshold; $('patience').value = m.patience; $('cooldown').value = m.cooldown;
      updateSliderLabels();
      $('recInfo').textContent = `录音目录：${m.record_dir}　已有 正样本 ${m.counts.positive} · 负样本 ${m.counts.negative} · 命中片段 ${m.counts.hit}`;
      setStatus(`已连接，模型：${m.models.join(', ')}`);
    } else if (m.type === 'frame') {
      state.frames++;
      state.lastScore = m.score;
      state.points.push({ t: m.t, s: m.score });
      const cutoff = m.t - WINDOW_SECONDS - 1;
      while (state.points.length && state.points[0].t < cutoff) state.points.shift();
      while (state.hitPoints.length && state.hitPoints[0].t < cutoff) state.hitPoints.shift();
      $('score').textContent = m.score.toFixed(2);
      // 同时加载了多个模型时，分别列出各自的分数（上面的大数字与曲线取最高分）
      if (m.scores && Object.keys(m.scores).length > 1) {
        $('perModel').textContent = Object.entries(m.scores)
          .map(([k, v]) => `${k.replace(/^siu_maau_jan_?/, '') || k} ${v.toFixed(2)}`).join('　');
      }
      $('fill').style.width = `${Math.min(100, m.score * 100)}%`;
      $('fill').classList.toggle('hit', m.score >= state.threshold);
      $('level').textContent = `${m.level_db.toFixed(0)} dBFS`;
      for (const d of m.fired) onHit(d, m.t);
    } else if (m.type === 'saved') {
      $('recInfo').textContent = `已保存 ${m.path}　正样本 ${m.counts.positive} · 负样本 ${m.counts.negative} · 命中片段 ${m.counts.hit}`;
    } else if (m.type === 'config') {
      state.threshold = m.threshold;
    } else if (m.type === 'error') {
      setStatus(`服务端错误：${m.message}`);
    }
  }

  function onHit(d, t) {
    state.hits++;
    state.hitPoints.push({ t, s: d.score });
    $('hits').textContent = state.hits;
    const tr = document.createElement('tr');
    const now = new Date();
    tr.innerHTML = `<td>${state.hits}</td><td>${now.toLocaleTimeString()}</td><td>${d.t.toFixed(2)} s</td>` +
                   `<td>${d.score.toFixed(3)}</td><td>${d.model}</td>`;
    $('log').prepend(tr);
    if ($('autoSave').checked) {
      clearTimeout(state.pendingAutoSave);
      state.pendingAutoSave = setTimeout(() => send({ type: 'save', label: 'hit', seconds: 3 }), 700);
    }
  }

  // ---------------------------------------------------------------- 麦克风
  async function start() {
    try {
      $('start').disabled = true;
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
      $('sr').textContent = `${ctx.sampleRate} Hz${ctx.sampleRate === 16000 ? '' : '（页面内重采样）'}`;
      const src = ctx.createMediaStreamSource(state.stream);
      if (ctx.audioWorklet) {
        await ctx.audioWorklet.addModule('/static/pcm-worklet.js');
        const node = new AudioWorkletNode(ctx, 'pcm-capture', { processorOptions: { targetRate: 16000 } });
        node.port.onmessage = (e) => { if (state.ws && state.ws.readyState === 1) state.ws.send(e.data); };
        src.connect(node);
        state.node = node;
      } else {
        state.node = scriptProcessorFallback(ctx, src);
      }
      state.running = true;
      state.startedAt = performance.now();
      $('stop').disabled = false;
      setStatus('监听中，对着麦克风说「小貓人」');
      send({ type: 'reset' });
      sendConfig();
    } catch (e) {
      setStatus(`无法开始：${e.message || e}`);
      $('start').disabled = false;
    }
  }

  // 没有 AudioWorklet 的老浏览器：ScriptProcessor + 主线程线性重采样
  function scriptProcessorFallback(ctx, src) {
    const node = ctx.createScriptProcessor(4096, 1, 1);
    const ratio = ctx.sampleRate / 16000;
    let buf = new Int16Array(1280), n = 0, pos = 0, prev = 0;
    node.onaudioprocess = (e) => {
      const ch = e.inputBuffer.getChannelData(0);
      let x = pos;
      while (x < ch.length) {
        const i = Math.floor(x), frac = x - i;
        const a = i < 0 ? prev : ch[i], b = i + 1 < ch.length ? ch[i + 1] : ch[ch.length - 1];
        const v = Math.max(-1, Math.min(1, a + (b - a) * frac));
        buf[n++] = v < 0 ? v * 32768 : v * 32767;
        if (n === 1280) { if (state.ws && state.ws.readyState === 1) state.ws.send(buf.buffer.slice(0)); n = 0; }
        x += ratio;
      }
      pos = x - ch.length; prev = ch[ch.length - 1];
    };
    src.connect(node); node.connect(ctx.destination);
    return node;
  }

  function stop() {
    state.running = false;
    if (state.node) { try { state.node.disconnect(); } catch (e) {} state.node = null; }
    if (state.stream) { state.stream.getTracks().forEach((t) => t.stop()); state.stream = null; }
    if (state.ctx) { state.ctx.close(); state.ctx = null; }
    $('start').disabled = false; $('stop').disabled = true;
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
  function updateSliderLabels() {
    $('thrV').textContent = Number($('threshold').value).toFixed(2);
    $('patV').textContent = $('patience').value;
    $('cdV').textContent = Number($('cooldown').value).toFixed(1);
    state.threshold = Number($('threshold').value);
    $('thrLine').style.left = `${state.threshold * 100}%`;
  }
  function sendConfig() {
    send({ type: 'config', threshold: Number($('threshold').value), patience: Number($('patience').value),
           cooldown: Number($('cooldown').value) });
  }
  for (const id of ['threshold', 'patience', 'cooldown']) {
    $(id).addEventListener('input', () => { updateSliderLabels(); sendConfig(); });
  }
  $('start').onclick = start;
  $('stop').onclick = stop;
  $('savePos').onclick = () => send({ type: 'save', label: 'positive', seconds: 3 });
  $('saveNeg').onclick = () => send({ type: 'save', label: 'negative', seconds: 3 });
  $('reset').onclick = () => { send({ type: 'reset' }); state.points = []; state.hitPoints = []; };
  function setStatus(s) { $('status').textContent = s; }

  // ---------------------------------------------------------------- 图
  const canvas = $('chart'); const g = canvas.getContext('2d');
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  let hoverX = null;
  canvas.addEventListener('mousemove', (e) => { hoverX = e.offsetX * (canvas.width / canvas.clientWidth); });
  canvas.addEventListener('mouseleave', () => { hoverX = null; $('hover').textContent = '最近 20 秒的分数；虚线是阈值，绿点是命中。鼠标悬停可读数。'; });

  function draw() {
    const W = canvas.width, H = canvas.height, padL = 34, padR = 10, padT = 10, padB = 18;
    g.clearRect(0, 0, W, H);
    const now = state.points.length ? state.points[state.points.length - 1].t : 0;
    const t0 = now - WINDOW_SECONDS;
    const x = (t) => padL + ((t - t0) / WINDOW_SECONDS) * (W - padL - padR);
    const y = (s) => padT + (1 - s) * (H - padT - padB);
    // 网格与刻度（低调）
    g.strokeStyle = css('--border'); g.lineWidth = 1; g.fillStyle = css('--muted'); g.font = '11px system-ui';
    for (const s of [0, 0.5, 1]) {
      g.beginPath(); g.moveTo(padL, y(s)); g.lineTo(W - padR, y(s)); g.stroke();
      g.fillText(s.toFixed(1), 4, y(s) + 4);
    }
    for (let t = Math.ceil(t0 / 5) * 5; t <= now; t += 5) {
      if (t < t0) continue;
      g.fillText(`${t.toFixed(0)}s`, x(t) - 8, H - 4);
    }
    // 阈值
    g.setLineDash([4, 4]); g.strokeStyle = css('--text-2');
    g.beginPath(); g.moveTo(padL, y(state.threshold)); g.lineTo(W - padR, y(state.threshold)); g.stroke();
    g.setLineDash([]);
    // 分数曲线
    if (state.points.length > 1) {
      g.strokeStyle = css('--series'); g.lineWidth = 2; g.lineJoin = 'round';
      g.beginPath();
      state.points.forEach((p, i) => { i ? g.lineTo(x(p.t), y(p.s)) : g.moveTo(x(p.t), y(p.s)); });
      g.stroke();
    }
    // 命中
    g.fillStyle = css('--good');
    for (const h of state.hitPoints) { g.beginPath(); g.arc(x(h.t), y(h.s), 5, 0, Math.PI * 2); g.fill(); }
    // 悬停读数
    if (hoverX !== null && state.points.length) {
      const tHover = t0 + ((hoverX - padL) / (W - padL - padR)) * WINDOW_SECONDS;
      let best = state.points[0];
      for (const p of state.points) if (Math.abs(p.t - tHover) < Math.abs(best.t - tHover)) best = p;
      g.strokeStyle = css('--muted'); g.lineWidth = 1;
      g.beginPath(); g.moveTo(x(best.t), padT); g.lineTo(x(best.t), H - padB); g.stroke();
      $('hover').textContent = `t = ${best.t.toFixed(2)} s　分数 ${best.s.toFixed(3)}`;
    }
    // 统计
    if (state.running) {
      const sec = (performance.now() - state.startedAt) / 1000;
      $('elapsed').textContent = `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, '0')}`;
      $('rate').textContent = sec > 30 ? (state.hits / sec * 3600).toFixed(1) : '（30 s 后统计）';
    }
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
  updateSliderLabels();
  if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) fillDevices().catch(() => {});
})();
