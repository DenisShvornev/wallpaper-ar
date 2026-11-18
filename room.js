// Модель комнаты из кадров сегментации: каждая стена — вертикальная плоскость над линией стыка с полом.
// Наблюдения с разных кадров сливаются, концы стен привязываются к углам (пересечениям соседних стен).
// Чистый JS без DOM/three.js — тестируется в node.
import {heightsFrom, median} from './geom.js?v=10';

export const COLORS = [0x38bdf8, 0xf472b6, 0xa3e635, 0xfb923c, 0xc084fc, 0x2dd4bf, 0xfacc15, 0xf87171];
const MERGE_ANG = Math.cos(10 * Math.PI / 180), MERGE_OFF = 0.12, MERGE_GAP = 0.8, MAXP = 800, MAXH = 300;
const CORNER_REACH = 0.6;   // насколько угол может быть за видимым краем стены, м

// PCA-прямая по точкам {x, z}
export function fitLine(P){
  let mx = 0, mz = 0; for (const q of P){ mx += q.x; mz += q.z; } mx /= P.length; mz /= P.length;
  let sxx = 0, sxz = 0, szz = 0; for (const q of P){ const x = q.x - mx, z = q.z - mz; sxx += x*x; sxz += x*z; szz += z*z; }
  const ang = 0.5 * Math.atan2(2*sxz, sxx - szz);
  return {mx, mz, dx: Math.cos(ang), dz: Math.sin(ang)};
}
// Несколько прямых последовательным RANSAC (фронтальная стена + боковые)
export function extractLines(pts, {maxLines = 3, minPts = 10, tol = 0.035, iters = 150, rnd = Math.random} = {}){
  let rest = pts, out = [];
  while (out.length < maxLines && rest.length >= minPts){
    let bestN = 0, best = null;
    for (let it = 0; it < iters; it++){
      const a = rest[rnd()*rest.length | 0], b = rest[rnd()*rest.length | 0];
      const dx = b.x - a.x, dz = b.z - a.z, len = Math.hypot(dx, dz); if (len < 0.2) continue;
      const nx = -dz/len, nz = dx/len; let c = 0;
      for (const q of rest) if (Math.abs((q.x - a.x)*nx + (q.z - a.z)*nz) < tol) c++;
      if (c > bestN){ bestN = c; best = {a, nx, nz}; }
    }
    if (!best || bestN < minPts) break;
    const inl = [], out2 = [];
    for (const q of rest) (Math.abs((q.x - best.a.x)*best.nx + (q.z - best.a.z)*best.nz) < tol ? inl : out2).push(q);
    out.push({...fitLine(inl), pts: inl}); rest = out2;
  }
  return out;
}

export class Room {
  constructor(){ this.walls = []; this.seq = 0; this.ver = 0; this.cams = []; this._c = {}; }
  clear(){ this.walls = []; this.cams = []; this.ver++; }
  // A — результат analyze() (точки уже в мировых координатах), groundY — уровень пола
  ingest(A, groundY){
    this.ver++;
    if (this.cams.length < 300) this.cams.push([A.cam[0], A.cam[2]]); else this.cams[Math.random()*300 | 0] = [A.cam[0], A.cam[2]];
    const lines = extractLines(A.pts);
    for (const L of lines){
      const hs = heightsFrom(A, L.mx, L.mz, L.dx, L.dz, groundY, new Set(L.pts.map(p => p.c)));
      const w = this._match(L) || this._add(L);
      for (const p of L.pts){ if (w.pts.length < MAXP) w.pts.push({x: p.x, z: p.z}); else w.pts[Math.random()*MAXP | 0] = {x: p.x, z: p.z}; }
      for (const h of hs){ if (w.hs.length < MAXH) w.hs.push(h); else w.hs[Math.random()*MAXH | 0] = h; }
      w.obs++; this._refit(w);
    }
    this._mergeAll();
    this.ver++; this._corners(this.visible());   // углы — только между настоящими стенами, не мебелью
    this.ver++;
    return lines.length;
  }
  _add(L){ const w = {id: ++this.seq, color: COLORS[this.seq % COLORS.length], ...L, pts: [], hs: [], obs: 0, t0: 0, t1: 0, end: [null, null]}; this.walls.push(w); return w; }
  _near(w, L){
    if (Math.abs(w.dx*L.dx + w.dz*L.dz) < MERGE_ANG) return false;
    if (Math.abs((L.mx - w.mx)*(-w.dz) + (L.mz - w.mz)*w.dx) > MERGE_OFF) return false;
    // по длине прямой отрезки должны перекрываться или почти касаться
    const tl = L.pts ? L.pts.map(p => (p.x - w.mx)*w.dx + (p.z - w.mz)*w.dz) : [(L.mx - w.mx)*w.dx + (L.mz - w.mz)*w.dz];
    const lo = Math.min(...tl), hi = Math.max(...tl);
    return lo < w.t1 + MERGE_GAP && hi > w.t0 - MERGE_GAP;
  }
  _match(L){ let best = null, bd = Infinity; for (const w of this.walls){ if (!this._near(w, L)) continue; const d = Math.abs((L.mx - w.mx)*(-w.dz) + (L.mz - w.mz)*w.dx); if (d < bd){ bd = d; best = w; } } return best; }
  _refit(w){
    const f = fitLine(w.pts);
    if (f.dx*w.dx + f.dz*w.dz < 0){ f.dx = -f.dx; f.dz = -f.dz; }   // сохраняем направление
    Object.assign(w, f);
    const t = Float64Array.from(w.pts, p => (p.x - w.mx)*w.dx + (p.z - w.mz)*w.dz).sort();
    w.t0 = t[Math.floor(t.length * 0.02)]; w.t1 = t[Math.min(t.length - 1, Math.floor(t.length * 0.98))];
  }
  _mergeAll(){
    for (let i = 0; i < this.walls.length; i++) for (let j = i + 1; j < this.walls.length; j++){
      const a = this.walls[i], b = this.walls[j];
      if (!this._near(a, {...b, pts: b.pts})) continue;
      const [keep, drop] = a.pts.length >= b.pts.length ? [a, b] : [b, a];
      keep.pts.push(...drop.pts.slice(0, MAXP - keep.pts.length)); keep.hs.push(...drop.hs.slice(0, MAXH - keep.hs.length)); keep.obs += drop.obs;
      this._refit(keep); this.walls.splice(this.walls.indexOf(drop), 1); j = i;   // заново проверяем пары с изменённой стеной
    }
  }
  _corners(list){
    for (const w of this.walls) w.end = [null, null];
    for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++){
      const a = list[i], b = list[j], cr = a.dx*b.dz - a.dz*b.dx;
      if (Math.abs(cr) < 0.7) continue;                                       // угол между стенами > ~45°
      const s = ((b.mx - a.mx)*b.dz - (b.mz - a.mz)*b.dx) / cr, x = a.mx + a.dx*s, z = a.mz + a.dz*s;
      for (const w of [a, b]){
        const t = (x - w.mx)*w.dx + (z - w.mz)*w.dz;
        if (t < w.t0 - CORNER_REACH || t > w.t1 + CORNER_REACH) continue;
        const k = Math.abs(t - w.t0) < Math.abs(t - w.t1) ? 0 : 1;
        if (!w.end[k] || Math.abs(t - [w.t0, w.t1][k]) < Math.abs(w.end[k].t - [w.t0, w.t1][k])) w.end[k] = {t, x, z};
      }
    }
  }
  // Кэш до следующего ingest(): кадр отрисовки не делает ни одной сортировки.
  _cache(key, f){ const c = this._c; if (c.ver !== this.ver){ this._c = {ver: this.ver}; } return key in this._c ? this._c[key] : (this._c[key] = f()); }
  roomHeight(){ return this._cache('rh', () => { const all = this.walls.flatMap(w => w.hs); return all.length >= 5 ? median(all) : null; }); }
  center(){ return this._cache('ctr', () => { if (!this.cams.length) return null; let x = 0, z = 0; for (const c of this.cams){ x += c[0]; z += c[1]; } return [x / this.cams.length, z / this.cams.length]; }); }
  // Размер стены: концы — углы, если найдены, иначе видимый край
  size(w){ return this._cache('s' + w.id, () => this._size(w)); }
  _size(w){
    const e0 = w.end[0] ? w.end[0].t : w.t0, e1 = w.end[1] ? w.end[1].t : w.t1;
    const h = w.hs.length >= 5 ? median(w.hs) : this.roomHeight();
    return {e0, e1, w: e1 - e0, h, full: !!(w.end[0] && w.end[1]), ownH: w.hs.length >= 5};
  }
  // Видимые стены. Линия, параллельная более крупной стене и стоящая перед ней (ближе к месту съёмки)
  // на 12–90 см с перекрытием по длине, — это низ мебели или плинтус, а не стена: скрываем.
  visible(){ return this._cache('vis', () => {
    const base = this.walls.filter(w => w.obs >= 2 && w.pts.length >= 20 && w.t1 - w.t0 > 0.25), c = this.center();
    if (!c) return base;
    const dist = w => Math.abs((c[0] - w.mx)*(-w.dz) + (c[1] - w.mz)*w.dx);
    return base.filter(w => !base.some(o => {
      if (o === w || o.pts.length < w.pts.length * 0.5 || Math.abs(o.dx*w.dx + o.dz*w.dz) < MERGE_ANG) return false;
      const off = Math.abs((w.mx - o.mx)*(-o.dz) + (w.mz - o.mz)*o.dx); if (off < MERGE_OFF || off > 0.9) return false;
      const a0 = (w.mx - o.mx)*o.dx + (w.mz - o.mz)*o.dz, lo = a0 + Math.min(w.t0, w.t1)*(w.dx*o.dx + w.dz*o.dz), hi = a0 + Math.max(w.t0, w.t1)*(w.dx*o.dx + w.dz*o.dz);
      return Math.max(Math.min(lo, hi), o.t0) < Math.min(Math.max(lo, hi), o.t1) && dist(o) > dist(w);
    }));
  }); }
  // Стена под прицелом: пересечение луча (o, d) с вертикальными плоскостями стен
  pick(o, d, floorY){
    let best = null, bt = Infinity;
    for (const w of this.visible()){
      const nx = -w.dz, nz = w.dx, den = d[0]*nx + d[2]*nz; if (Math.abs(den) < 1e-3) continue;
      const t = ((w.mx - o[0])*nx + (w.mz - o[2])*nz) / den; if (t <= 0 || t >= bt) continue;
      const x = o[0] + d[0]*t, y = o[1] + d[1]*t, z = o[2] + d[2]*t, s = (x - w.mx)*w.dx + (z - w.mz)*w.dz, S = this.size(w);
      if (s < S.e0 - 0.05 || s > S.e1 + 0.05 || y < floorY - 0.05 || y > floorY + (S.h || 3) + 0.05) continue;
      best = w; bt = t;
    }
    return best;
  }
}
