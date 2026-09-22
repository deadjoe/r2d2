// A stateful box-filter resampler: sample boundaries persist across render quanta.
// The main graph also low-passes input before downsampling from 44.1/48 kHz.
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.left = this.ratio;
    this.sum = 0;
    this.frame = new Int16Array(2560);
    this.offset = 0;
    this.stopped = false;
    this.port.onmessage = ({data}) => {
      if (data === 'stop') {
        this.stopped = true;
        if (this.offset) this.port.postMessage(this.frame.slice(0, this.offset).buffer);
        this.port.postMessage('flushed');
      }
    };
  }
  process(inputs) {
    if (this.stopped) return false;
    const channels = inputs[0];
    if (!channels?.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let value = 0;
      for (const channel of channels) value += channel[i] / channels.length;
      let available = 1;
      while (available > 1e-9) {
        const take = Math.min(available, this.left);
        this.sum += value * take;
        this.left -= take;
        available -= take;
        if (this.left < 1e-9) {
          const sample = Math.max(-1, Math.min(32767 / 32768, this.sum / this.ratio));
          this.frame[this.offset++] = Math.round(sample * 32768);
          this.left = this.ratio;
          this.sum = 0;
          if (this.offset === 2560) {
            this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
            this.frame = new Int16Array(2560);
            this.offset = 0;
          }
        }
      }
    }
    return true;
  }
}
registerProcessor('pcm-capture', PCMProcessor);
