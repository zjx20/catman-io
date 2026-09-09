// AudioWorklet：把麦克风的 float32 采样重采样到 16 kHz，攒够 1280 个（80 ms）转 int16 发回主线程。
// 浏览器一般会尊重 AudioContext({sampleRate: 16000})，那样 ratio 就是 1；不尊重时做线性插值重采样。
class PCMCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const target = (options.processorOptions && options.processorOptions.targetRate) || 16000;
    this.ratio = sampleRate / target;
    this.frameSize = 1280;
    this.buf = new Int16Array(this.frameSize);
    this.n = 0;
    this.pos = 0; // 下一个输出样本在当前输入块里的位置（可为负：落在上一块）
    this.prev = 0;
  }

  push(v) {
    const s = Math.max(-1, Math.min(1, v));
    this.buf[this.n++] = s < 0 ? s * 32768 : s * 32767;
    if (this.n === this.frameSize) {
      this.port.postMessage(this.buf.buffer.slice(0));
      this.n = 0;
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];
    if (this.ratio === 1) {
      for (let i = 0; i < ch.length; i++) this.push(ch[i]);
      return true;
    }
    let x = this.pos;
    while (x < ch.length) {
      const i = Math.floor(x);
      const frac = x - i;
      const a = i < 0 ? this.prev : ch[i];
      const b = i + 1 < ch.length ? ch[i + 1] : ch[ch.length - 1];
      this.push(a + (b - a) * frac);
      x += this.ratio;
    }
    this.pos = x - ch.length;
    this.prev = ch[ch.length - 1];
    return true;
  }
}

registerProcessor('pcm-capture', PCMCapture);
