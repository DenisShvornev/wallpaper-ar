// Фоновый поток: нейросеть SegFormer-B0 (ADE20K) не блокирует AR-рендер.
import {AutoModel, Tensor, env} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
env.allowLocalModels = false;
let model = null;
self.onmessage = async ({data: m}) => {
  try {
    if (m.type === 'load'){
      if (!model){
        const files = {};
        model = await AutoModel.from_pretrained('Xenova/segformer-b0-finetuned-ade-512-512', {dtype: 'q8', device: 'wasm',
          progress_callback: p => {
            if (p.status === 'progress' && p.total){
              files[p.file] = [p.loaded, p.total]; let l = 0, t = 0;
              for (const k in files){ l += files[k][0]; t += files[k][1]; }
              self.postMessage({type: 'progress', pct: Math.round(l / t * 100)});
            }
          }});
      }
      self.postMessage({type: 'ready'});
    } else if (m.type === 'run'){
      const t0 = performance.now();
      const out = await model({pixel_values: new Tensor('float32', m.data, [1, 3, m.H, m.W])});
      const logits = out.logits.data;
      self.postMessage({type: 'result', id: m.id, logits, dims: out.logits.dims, ms: Math.round(performance.now() - t0)}, [logits.buffer]);
    }
  } catch (e){
    self.postMessage({type: 'error', id: m.id, message: String(e && e.message || e)});
  }
};
