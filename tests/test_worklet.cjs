// Exercise the actual AudioWorklet resampler without opening a microphone.
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = readFileSync('web/capture-worklet.js', 'utf8');
for (const rate of [16000, 44100, 48000]) {
  let Processor;
  const messages = [];
  class AudioWorkletProcessor {
    constructor() { this.port = {postMessage: data => messages.push(data)}; }
  }
  vm.runInNewContext(source, {AudioWorkletProcessor, sampleRate: rate,
    registerProcessor: (name, klass) => Processor = klass, Int16Array, Math});
  const capture = new Processor();
  for (let pos = 0; pos < rate; pos += 128) {
    capture.process([[new Float32Array(Math.min(128, rate - pos)).fill(.25)]]);
  }
  capture.port.onmessage({data:'stop'});
  assert.equal(messages.at(-1), 'flushed');
  const frames = messages.slice(0, -1).map(x => new Int16Array(x));
  assert.equal(frames.reduce((n, frame) => n + frame.length, 0), 16000);
  assert.ok(frames.every(frame => frame.length <= 2560));
  assert.ok(frames.every(frame => frame.every(sample => sample === 8192)));
  assert.equal(capture.process([]), false);
}
console.log('AudioWorklet: 16 / 44.1 / 48 kHz packet boundaries and stop flush passed');
