/* policy.js — a trained agent's brain, run in the browser.

   Exactly what evaluate.Agent does in Python, for the PPO agents A and B:
     1. normalize the 24 numbers the robot sees with the stats saved during training
        ((x - mean) / sqrt(var + eps), clipped to +-10),
     2. run the policy network: 24 -> 64 -> 64 (tanh) -> 4,
     3. take the "average" move (deterministic) and clip it to [-1, 1].
   The network is float32 in PyTorch; every layer's output is rounded to float32 here too. */

const f32 = Math.fround;

function decode(b64, Type) {
  const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

export class Brain {
  constructor(spec) {
    this.mean = decode(spec.obs_mean, Float64Array);
    this.var = decode(spec.obs_var, Float64Array);
    this.eps = spec.epsilon;
    this.clip = spec.clip_obs;
    this.layers = spec.layers.map(l => ({
      rows: l.rows, cols: l.cols, tanh: l.activation === "tanh",
      w: decode(l.weight, Float32Array), b: decode(l.bias, Float32Array),
    }));
    this.scale = Float64Array.from(this.var, v => Math.sqrt(v + this.eps));
  }

  act(obs) {
    let x = new Float32Array(obs.length);
    for (let i = 0; i < obs.length; i++) {
      const v = (obs[i] - this.mean[i]) / this.scale[i];
      x[i] = f32(Math.min(this.clip, Math.max(-this.clip, v)));
    }
    for (const L of this.layers) {
      const y = new Float32Array(L.rows);
      for (let r = 0; r < L.rows; r++) {
        let s = L.b[r];
        const off = r * L.cols;
        for (let c = 0; c < L.cols; c++) s += L.w[off + c] * x[c];
        y[r] = L.tanh ? f32(Math.tanh(f32(s))) : f32(s);
      }
      x = y;
    }
    for (let i = 0; i < x.length; i++) x[i] = Math.min(1, Math.max(-1, x[i]));
    return x;
  }
}
