// Фоновый поток: SegFormer-B0 (ADE20K) + разбор кадра. Главный поток получает только готовые точки и маску.
// WebGPU (если есть) в разы быстрее WASM; при любой ошибке WebGPU — автоматический откат на WASM q8.
import {AutoModel, Tensor, env} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
import {analyze} from './geom.js?v=8';
env.allowLocalModels = false;

const MODEL = 'Xenova/segformer-b0-finetuned-ade-512-512';
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
let model = null, device = '', loading = null;
const files = {};
const progress = p => {
  if (p.status !== 'progress' || !p.total) return;
  files[p.file] = [p.loaded, p.total]; let l = 0, t = 0;
  for (const k in files){ l += files[k][0]; t += files[k][1]; }
  self.postMessage({type: 'progress', pct: Math.round(l / t * 100)});
};
function toTensor(rgba, W, H){
  const N = W*H, d = new Float32Array(3*N);
  const a0 = 1/(255*STD[0]), a1 = 1/(255*STD[1]), a2 = 1/(255*STD[2]), b0 = MEAN[0]/STD[0], b1 = MEAN[1]/STD[1], b2 = MEAN[2]/STD[2];
  for (let o = 0, i = 0; o < N; o++, i += 4){ d[o] = rgba[i]*a0 - b0; d[N + o] = rgba[i+1]*a1 - b1; d[2*N + o] = rgba[i+2]*a2 - b2; }
  return new Tensor('float32', d, [1, 3, H, W]);
}
async function tryLoad(opts){
  const m = await AutoModel.from_pretrained(MODEL, {...opts, progress_callback: progress});
  await m({pixel_values: toTensor(new Uint8Array(288*512*4), 288, 512)}); // прогрев: компиляция ядер до первого кадра
  return m;
}
async function load(){
  const cands = [];
  try { if (self.navigator?.gpu && await navigator.gpu.requestAdapter()) cands.push({device: 'webgpu', dtype: 'fp32'}); } catch {}
  cands.push({device: 'wasm', dtype: 'q8'});
  let err;
  for (const c of cands){
    try { model = await tryLoad(c); device = c.device; return; }
    catch (e){ err = e; for (const k in files) delete files[k]; }
  }
  throw err;
}
self.onmessage = async ({data: m}) => {
  try {
    if (m.type === 'load'){
      loading ??= load();
      await loading;
      self.postMessage({type: 'ready', device});
    } else if (m.type === 'run'){
      if (!model) throw new Error('модель не загружена');
      const t0 = performance.now();
      const out = await model({pixel_values: toTensor(m.rgba, m.W, m.H)});
      const [, C, Hl, Wl] = out.logits.dims;
      const A = analyze({logits: out.logits.data, C, Hl, Wl, camW: m.camW, projInv: m.projInv, groundY: m.groundY});
      self.postMessage({type: 'result', id: m.id, ...A, ms: Math.round(performance.now() - t0)}, [A.lab.buffer]);
    }
  } catch (e){
    if (m.type === 'load') loading = null;
    self.postMessage({type: 'error', id: m.id, stage: m.type, message: String(e && e.message || e)});
  }
};
