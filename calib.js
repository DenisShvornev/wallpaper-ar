// Поиск пола «на лету» — без отдельного шага калибровки для пользователя.
//
// Пользователь просто наводит камеру на пол. Как только движок отдаёт трекинг NORMAL и есть
// несколько согласованных хитов плоскости ниже телефона (обычно < 1 с), пол готов и появляется прицел.
//  • пол = самый нижний плотный кластер хитов (±3 см): стол/диван выше пола отбрасываются;
//  • 8th Wall: если хитов нет (hitTest часто пуст), берём y = 0 — в режиме absolute scale это уровень пола по определению движка;
//  • высота телефона над полом должна быть правдоподобной (0.6–2.2 м) — тихая проверка метрического масштаба;
//    пока 8th Wall уточняет масштаб, просто просим поводить телефоном.
// Дальше пол тихо уточняется по новым хитам, при дрейфе трекинга на 3–20 см переустанавливается сам.
// Потеря трекинга пол не сбрасывает: на это время главный модуль переходит в режим гироскопа.
import {floorCluster} from './geom.js?v=10';

export const CAL = {samples: 6, tol: 0.03, camH: [0.6, 2.2], lostReset: 3000};

export class Calibration {
  constructor(engine){ this.engine = engine; this.prior = engine === 'webxr' ? null : 0; this.reset(); }
  reset(msg = ''){
    Object.assign(this, {ok: false, msg, ys: [], lostSince: null, floorY: null, floor: null, camH: null, n: 0, trk: false, k: 0});
  }
  // cam: [x, y, z] позиция камеры; hitY: высота хита горизонтальной плоскости в центре экрана или null.
  // Возвращает событие: 'done' | 'lost' | 'reground' | null.
  update({now, tracking, cam, hitY}){
    // Плохим считаем только явный LIMITED/LOST: если движок не прислал статус, не зависаем навсегда.
    const trk = !/^(LIMITED|NOT_AVAILABLE|LOST)/.test(String(tracking || ''));
    const hy = hitY != null && hitY < cam[1] - 0.3 ? hitY : null;   // только поверхности заметно ниже телефона
    return this.ok ? this._watch(now, trk, cam, hy) : this._collect(now, trk, cam, hy);
  }
  _collect(now, trk, cam, hy){
    this.trk = trk;
    if (!trk) return null;
    if (hy != null){ this.ys.push(hy); if (this.ys.length > 60) this.ys.shift(); }
    const c = floorCluster(this.ys, CAL.tol, CAL.samples);
    this.n = c ? c.n : 0;
    this.floor = c && c.n >= CAL.samples ? c.y : this.prior;
    this.src = c && c.n >= CAL.samples ? 'hits' : this.prior != null ? 'y=0' : '';
    this.camH = this.floor != null ? cam[1] - this.floor : null;
    if (this.floor != null && this.hOk()){ this.ok = true; this.floorY = this.floor; this.k = 0; this.msg = ''; return 'done'; }
    return null;
  }
  _watch(now, trk, cam, hy){
    this.trk = trk;
    if (!trk || hy == null) return null;   // при сбое трекинга главный модуль сам переходит на гироскоп
    this.ys.push(hy); if (this.ys.length > 60) this.ys.shift();
    if (++this.k % 10) return null;
    const c = floorCluster(this.ys, CAL.tol, 20); if (!c || c.n < 20) return null;
    const dy = c.y - this.floorY, camH = cam[1] - c.y;
    if (Math.abs(dy) <= 0.03){ this.floorY += dy * 0.3; this.src = 'hits'; return null; }                   // тихое уточнение
    if (Math.abs(dy) < 0.2 && camH > CAL.camH[0] && camH < CAL.camH[1]){ this.floorY = c.y; this.src = 'hits'; this.ys = []; return 'reground'; }
    return null;                                                                        // > 20 см — это мебель, не пол
  }
  hOk(){ return this.camH != null && this.camH > CAL.camH[0] && this.camH < CAL.camH[1]; }
  // Одна короткая подсказка вместо чек-листа.
  hint(){
    const pre = this.msg ? this.msg + '. ' : '';
    if (!this.trk) return pre + 'Плавно поведите телефоном из стороны в сторону';
    if (this.floor == null) return pre + 'Наведите камеру на пол';
    return pre + 'Ещё немного поводите телефоном — уточняю масштаб';
  }
}
