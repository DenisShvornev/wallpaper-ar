// Калибровка пола и масштаба. Без DOM — тестируется в node.
//
// Что проверяем (всё параллельно, обычно укладывается в 2–4 с):
//  • трекинг NORMAL;
//  • пол: >= 12 реальных хитов плоскости ниже телефона, собранных в нижний плотный кластер (±2.5 см);
//    фиктивная плоскость y=0 не засчитывается, стол/диван выше пола отбрасываются;
//  • 8th Wall: телефон сместился на 40 см (нужно движку для абсолютного масштаба); WebXR/ARCore — не нужно;
//  • высота телефона над полом правдоподобна (0.6–2.2 м) — проверка, что масштаб метрический.
// После калибровки пол — плоскость y = floorY. Она тихо уточняется по новым хитам, а при дрейфе
// трекинга на 3–20 см переустанавливается без повторной калибровки. Сброс — только если трекинг потерян > 3 с.
import {floorCluster} from './geom.js?v=6';

export const CAL = {time: 1.5, samples: 12, tol: 0.025, camH: [0.6, 2.2], path8w: 0.4, lostReset: 3000};

export class Calibration {
  constructor(engine){ this.engine = engine; this.needPath = engine === 'webxr' ? 0 : CAL.path8w; this.reset(); }
  reset(msg = ''){
    Object.assign(this, {ok: false, msg, t0: null, time: 0, path: 0, last: null, ys: [], lostSince: null,
      floorY: null, floor: null, camH: null, n: 0, trk: false, progress: 0, k: 0});
  }
  // cam: [x, y, z] позиция камеры; hitY: высота хита горизонтальной плоскости в центре экрана или null.
  // Возвращает событие: 'done' | 'lost' | 'reground' | null.
  update({now, tracking, cam, hitY}){
    const trk = String(tracking || '').startsWith('NORMAL');
    const hy = hitY != null && hitY < cam[1] - 0.3 ? hitY : null;   // только поверхности заметно ниже телефона
    return this.ok ? this._watch(now, trk, cam, hy) : this._collect(now, trk, cam, hy);
  }
  _collect(now, trk, cam, hy){
    this.trk = trk;
    if (!trk){
      this.lostSince ??= now;
      if (now - this.lostSince > 2500){ this.t0 = null; this.path = 0; this.last = null; } // долгий сбой — движение заново, пол оставляем
      return null;
    }
    this.lostSince = null; this.t0 ??= now; this.time = (now - this.t0) / 1000;
    if (this.last){ const d = Math.hypot(cam[0] - this.last[0], cam[1] - this.last[1], cam[2] - this.last[2]); if (d < 0.25) this.path += d; }
    this.last = cam;
    if (hy != null){ this.ys.push(hy); if (this.ys.length > 90) this.ys.shift(); }
    const c = floorCluster(this.ys, CAL.tol, CAL.samples);
    this.n = c ? c.n : 0;
    this.floor = c && c.n >= CAL.samples ? c.y : null;
    this.camH = this.floor != null ? cam[1] - this.floor : null;
    const hOk = this.hOk();
    const pF = Math.min(1, this.n / CAL.samples), pP = this.needPath ? Math.min(1, this.path / this.needPath) : 1, pT = Math.min(1, this.time / CAL.time);
    this.progress = Math.min(hOk || this.floor == null ? 1 : 0.95, (pF*2 + pP*2 + pT) / 5);
    if (pF >= 1 && pP >= 1 && pT >= 1 && hOk){
      this.ok = true; this.floorY = this.floor; this.progress = 1; this.k = 0;
      return 'done';
    }
    return null;
  }
  _watch(now, trk, cam, hy){
    if (!trk){
      this.lostSince ??= now;
      if (now - this.lostSince > CAL.lostReset){ this.reset('трекинг был потерян'); return 'lost'; }
      return null;
    }
    this.lostSince = null;
    if (hy == null) return null;
    this.ys.push(hy); if (this.ys.length > 60) this.ys.shift();
    if (++this.k % 10) return null;
    const c = floorCluster(this.ys, CAL.tol, 20); if (!c || c.n < 20) return null;
    const dy = c.y - this.floorY, camH = cam[1] - c.y;
    if (Math.abs(dy) <= 0.03){ this.floorY += dy * 0.3; return null; }                   // тихое уточнение
    if (Math.abs(dy) < 0.2 && camH > CAL.camH[0] && camH < CAL.camH[1]){ this.floorY = c.y; this.ys = []; return 'reground'; }
    return null;                                                                        // > 20 см — это мебель, не пол
  }
  hOk(){ return this.camH != null && this.camH > CAL.camH[0] && this.camH < CAL.camH[1]; }
  hint(){
    const ok = b => b ? '✓' : '○', L = ['КАЛИБРОВКА' + (this.msg ? ' — ' + this.msg : '')];
    if (!this.trk){ L.push('○ Медленно поводите телефоном, глядя на пол'); return L.join('\n'); }
    const fDone = this.floor != null;
    L.push(ok(fDone) + ' Пол: ' + (fDone ? 'найден' : 'наведите камеру на пол (' + Math.min(this.n, CAL.samples) + '/' + CAL.samples + ')'));
    if (this.needPath) L.push(ok(this.path >= this.needPath) + ' Сделайте шаг в сторону: ' + Math.round(Math.min(this.path, this.needPath) * 100) + ' / ' + Math.round(this.needPath * 100) + ' см');
    if (fDone) L.push(ok(this.hOk()) + ' Высота телефона: ' + this.camH.toFixed(2) + ' м' + (this.hOk() ? '' : ' — неправдоподобно, нажмите ⟲'));
    return L.join('\n');
  }
}
