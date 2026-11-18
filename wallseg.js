// Замер стены по краям: калибровка -> левый угол -> правый угол -> потолок.
// Нейросеть SegFormer-B0 (ADE20K, Transformers.js, фоновый поток segworker.js) запускается только на неподвижном кадре
// с хорошим трекингом. Каждый найденный край закрепляется точкой в мире AR, накопления/расширения нет.
export const WALL = 0, CEIL = 5;
export const WALLISH = new Set([0, 1, 8, 14, 18, 22, 25, 27]); // wall, building, windowpane, door, curtain, painting, house, mirror
export const FLOORISH = new Set([3, 6, 11, 13, 28]);           // floor, road, sidewalk, earth, rug

// ---------- Чистая геометрия (без three.js — тестируется в node) ----------
const mul4 = (e, x, y, z, w) => [e[0]*x + e[4]*y + e[8]*z + e[12]*w, e[1]*x + e[5]*y + e[9]*z + e[13]*w, e[2]*x + e[6]*y + e[10]*z + e[14]*w, e[3]*x + e[7]*y + e[11]*z + e[15]*w];
export function rayDir(camW, projInv, u, v){
  let p = mul4(projInv, u*2 - 1, 1 - v*2, 0.5, 1); p = [p[0]/p[3], p[1]/p[3], p[2]/p[3]];
  const q = mul4(camW, p[0], p[1], p[2], 1);
  const d = [q[0] - camW[12], q[1] - camW[13], q[2] - camW[14]], n = Math.hypot(...d);
  return [d[0]/n, d[1]/n, d[2]/n];
}
const median = a => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };
function ransacLine(P, iters = 120, tol = 0.04){
  let best = null;
  for (let it = 0; it < iters; it++){
    const a = P[Math.random()*P.length | 0], b = P[Math.random()*P.length | 0];
    const dx = b.x - a.x, dz = b.z - a.z, len = Math.hypot(dx, dz); if (len < 0.15) continue;
    const inl = P.filter(q => Math.abs((q.x - a.x)*dz/len - (q.z - a.z)*dx/len) < tol);
    if (!best || inl.length > best.length) best = inl;
  }
  if (!best || best.length < 6) return null;
  let mx = 0, mz = 0; for (const q of best){ mx += q.x; mz += q.z; } mx /= best.length; mz /= best.length;
  let sxx = 0, sxz = 0, szz = 0; for (const q of best){ const x = q.x - mx, z = q.z - mz; sxx += x*x; sxz += x*z; szz += z*z; }
  const ang = 0.5 * Math.atan2(2*sxz, sxx - szz);
  const mc = best.reduce((s, q) => s + q.c, 0) / best.length;
  return {mx, mz, dx: Math.cos(ang), dz: Math.sin(ang), pts: best, mc};
}

// Разбор кадра: точки стыка «стена–пол» (в мире) и лучи на стык «стена–потолок» по столбцам
export function analyze({logits, C, Hl, Wl, camW, projInv, groundY}){
  const N = Hl * Wl, lab = new Uint8Array(N);
  for (let p = 0; p < N; p++){ let b = -1e9, bi = 0; for (let k = 0; k < C; k++){ const v = logits[k*N + p]; if (v > b){ b = v; bi = k; } } lab[p] = bi; }
  const L = (k, r, c) => logits[k*N + r*Wl + c];
  const cam = [camW[12], camW[13], camW[14]], pts = [], tops = {};
  for (let c = 0; c < Wl; c++){
    let r = Hl - 1;
    while (r > Hl * 0.55 && !FLOORISH.has(lab[r*Wl + c])) r--;
    if (FLOORISH.has(lab[r*Wl + c])){
      while (r >= 0 && FLOORISH.has(lab[r*Wl + c])) r--;
      if (r >= 1 && WALLISH.has(lab[r*Wl + c]) && WALLISH.has(lab[(r-1)*Wl + c])){
        const k = lab[r*Wl + c], fk = lab[(r+1)*Wl + c];
        const d0 = L(k, r, c) - L(fk, r, c), d1 = L(k, r+1, c) - L(fk, r+1, c);
        const t = d0 - d1 > 1e-6 ? Math.min(1, Math.max(0, d0 / (d0 - d1))) : 0.5;
        const d = rayDir(camW, projInv, (c + 0.5) / Wl, (r + 0.5 + t) / Hl);
        if (d[1] < -0.02){ const s = (groundY - cam[1]) / d[1]; if (s > 0 && s < 10) pts.push({c, x: cam[0] + d[0]*s, z: cam[2] + d[2]*s}); }
      }
    }
    if (lab[c] === CEIL){
      let r0 = 0; while (r0 < Hl && lab[r0*Wl + c] === CEIL) r0++;
      if (r0 < Hl - 1 && WALLISH.has(lab[r0*Wl + c])){
        const k = lab[r0*Wl + c];
        const dA = L(k, r0-1, c) - L(CEIL, r0-1, c), dB = L(k, r0, c) - L(CEIL, r0, c);
        const t = dB - dA > 1e-6 ? Math.min(1, Math.max(0, -dA / (dB - dA))) : 0.5;
        tops[c] = rayDir(camW, projInv, (c + 0.5) / Wl, (r0 - 0.5 + t) / Hl);
      }
    }
  }
  return {lab, Hl, Wl, pts, tops, cam, camW};
}

// Угол = пересечение двух прямых основания (фронтальная и боковая стена).
// which: 'L' — левый угол (фронтальная стена справа от него), 'R' — правый.
export function findCorner(A, which){
  if (A.pts.length < 12) return {error: 'nobase'};
  const l1 = ransacLine(A.pts); if (!l1) return {error: 'nobase'};
  const set1 = new Set(l1.pts);
  const l2 = ransacLine(A.pts.filter(q => !set1.has(q)));
  if (!l2) return {error: 'nocorner'};
  const cr = l1.dx*l2.dz - l1.dz*l2.dx;
  if (Math.abs(cr) < 0.7) return {error: 'nocorner'};               // стены должны сходиться под углом > ~45°
  const t = ((l2.mx - l1.mx)*l2.dz - (l2.mz - l1.mz)*l2.dx) / cr;
  const x = l1.mx + l1.dx*t, z = l1.mz + l1.dz*t;
  const near = l => Math.min(...l.pts.map(q => Math.hypot(q.x - x, q.z - z)));
  if (near(l1) > 1.2 || near(l2) > 1.2) return {error: 'nocorner'}; // угол далеко от видимых стыков — ненадёжно
  const [left, right] = l1.mc < l2.mc ? [l1, l2] : [l2, l1];
  const front = which === 'L' ? right : left;
  let dx = front.dx, dz = front.dz;
  if (dx*A.camW[0] + dz*A.camW[2] < 0){ dx = -dx; dz = -dz; }      // направление вдоль стены «вправо»
  return {x, z, dx, dz, frontCols: new Set(front.pts.map(q => q.c))};
}

// Высоты: лучи на стык с потолком пересекаются с вертикальной плоскостью стены
export function heightsFrom(A, px, pz, dx, dz, groundY, cols){
  const nx = -dz, nz = dx, hs = [];
  for (const c in A.tops){
    if (cols && !cols.has(+c)) continue;
    const d = A.tops[c], den = d[0]*nx + d[2]*nz; if (Math.abs(den) < 1e-4) continue;
    const t = ((px - A.cam[0])*nx + (pz - A.cam[2])*nz) / den; if (t <= 0) continue;
    const h = A.cam[1] + d[1]*t - groundY;
    if (h > 1.5 && h < 6) hs.push(h);
  }
  return hs;
}

// Высота по вертикали над точкой base (ручной режим): ближайшая точка между лучом камеры и вертикалью
export function heightOnVertical(bx, by, bz, o, d){
  const w0x = o[0] - bx, w0y = o[1] - by, w0z = o[2] - bz;
  const a = d[0]*d[0] + d[1]*d[1] + d[2]*d[2], b = d[1];
  const dd = d[0]*w0x + d[1]*w0y + d[2]*w0z, e = w0y, den = a - b*b;
  if (den < 1e-6) return null;
  if ((b*e - dd) / den <= 0) return null;
  return (a*e - b*dd) / den;
}

// ---------- Браузерная часть ----------
const STEADY_FRAMES = 12;
const S = {active: false, ready: false, worker: null, mode: 'idle', // idle | aim | busy | wait
  step: 'L', CL: null, CR: null, hs: [], status: '', fails: 0,
  gl: null, buf: null, groundY: 0, ms: 0, last: null, trk: null, prevPos: null, prevQ: null, steady: 0, pending: null,
  app: null, liveH: null,
  cal: null};
let ctx = null, quad, postL, postR, lineB, lineT, ring, thumb;
const newCal = () => ({ok: false, t0: performance.now(), path: 0, last: null, ys: [], hits: [], lostSince: null, jump: 0, chk: {}});
S.cal = newCal();

export function init(c){
  ctx = c; const {THREE, group} = c;
  quad = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({color: 0x38bdf8, transparent: true, opacity: 0.3, side: THREE.DoubleSide, depthTest: false, depthWrite: false}));
  quad.renderOrder = 9; quad.visible = false; group.add(quad);
  const bar = col => { const m = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, 1, 8), new THREE.MeshBasicMaterial({color: col, depthTest: false})); m.renderOrder = 11; m.visible = false; group.add(m); return m; };
  postL = bar(0x22c55e); postR = bar(0x22c55e); lineB = bar(0x22c55e); lineT = bar(0x22c55e);
  ring = new THREE.Mesh(new THREE.RingGeometry(0.035, 0.05, 32).rotateX(-Math.PI/2), new THREE.MeshBasicMaterial({color: 0xffffff, depthTest: false, transparent: true, opacity: .9, side: THREE.DoubleSide}));
  ring.renderOrder = 12; ring.visible = false; group.add(ring);
  S.prevPos = new THREE.Vector3(); S.prevQ = new THREE.Quaternion();
  const h = document.getElementById('hint'); if (h) h.style.whiteSpace = 'pre-line';
  if (!thumb){
    thumb = document.createElement('canvas');
    thumb.style.cssText = 'position:absolute;right:8px;top:calc(110px + env(safe-area-inset-top));width:64px;border:1px solid #fff6;border-radius:4px;image-rendering:pixelated;pointer-events:none;display:none';
    document.getElementById('hud').appendChild(thumb);
  }
}
export const setGround = y => { S.groundY = y; };
export const isCalibrated = () => S.cal.ok;

// ---------- Обязательная калибровка ----------
// Пройдена, когда: трекинг NORMAL >= 4 с, телефон прошёл >= 1 м, пол найден и стабилен (разброс < 1.5 см),
// высота телефона над полом правдоподобна (0.8–2.1 м — проверка абсолютного масштаба).
export function calibUpdate(cam, st){
  S.app = st;
  const C = S.cal, now = performance.now();
  const trkOk = String(st.tracking || '').startsWith('NORMAL');
  const floorHit = st.hit && (st.hitType === 'DETECTED_SURFACE' || st.hitType === 'ESTIMATED_SURFACE' || st.hitType === 'plane' || st.hitType === 'y=0');
  const e = cam.matrixWorld.elements, pos = [e[12], e[13], e[14]];
  if (C.ok){
    if (!trkOk){ C.lostSince ??= now; if (now - C.lostSince > 1500) recalibrate('Трекинг потерян — пройдите калибровку заново'); }
    else C.lostSince = null;
    if (floorHit){ C.ys.push(st.hit.y); if (C.ys.length > 30) C.ys.shift(); } if (C.ys.length >= 20 && Math.abs(median(C.ys) - C.fl; } else C.jump = 0;
    return C.ok;
  }
  if (!trkOk){ C.lostSince ??= now; if (now - C.lostSince > 2000) Object.assign(C, newCal(), {chk: {trk: false}}); else C.chk.trk
  if (C.last){ const d = Math.hypot(pos[0] - C.last[0], pos[1] - C.last[1], pos[2] - C.last[2]); if (d < 0.2) C.path += d; }
  C.last = pos;
  C.hits.push(floorHit ? 1 : 0); if (C.hits.length > 90) C.hits.shift();
  if (floorHit){ C.ys.push(st.hit.y); if (C.ys.length > 60) C.ys.shift(); }
  const rate = C.hits.reduce((a, b) => a + b, 0) / C.hits.length;
  const fy = median(C.ys) ?? 0, mean = C.ys.reduce((a, b) => a + b, 0) / (C.ys.length || 1);
  const sd = median(C.ys.map(y => Math.abs(y - fy))) ?? 1;
  const camH = pos[1] - fy;
  C.chk = {trk: true, time: (now - C.t0) / 1000, path: C.path, floor: C.floorOk = C.floorOk || (C.ys.length >= 20 && rate >= 0.3 && sd < 0.03), camH, n: C.ys.length};
  if (C.chk.time >= 3 && C.path >= 0.7 && C.chk.floor && camH > 0.8 && camH < 2.1){
    C.ok = true; C.floorY = fy; S.groundY = fy; S.status = '';
    if (navigator.vibrate) navigator.vibrate([30, 60, 30]);
  }
  return C.ok;
}
export function recalibrate(msg){
  S.cal = newCal(); S.cal.msg = msg || '';
  resetWall();
}
function calibHint(){
  const c = S.cal.chk, ok = b => b ? '✓' : '○';
  const lines = ['КАЛИБРОВКА (обязательно)' + (S.cal.msg ? ': ' + S.cal.msg : '')];
  if (!c.trk){ lines.push('○ Медленно водите телефоном по полу'); return lines.join('\n'); }
  lines.push(ok(c.floor) + ' Пол найден' + (c.floor ? '' : ' — наведите телефон на пол (' + Math.min(20, c.n || 0) + '/20)'));
  lines.push(ok(c.path >= 0.7) + ' Пройдите с телефоном ~0.7 м: ' + Math.min(0.7, c.path).toFixed(2) + ' / 0.70 м');
  const hOk = c.camH > 0.8 && c.camH < 2.1;
  lines.push(ok(hOk) + ' Высота телефона над полом: ' + (c.floor ? c.camH.toFixed(2) + ' м' : '—') + (c.floor && !hOk ? ' (неправдоподобно — сбросьте ⟲)' : ''));
  lines.push(ok(c.time >= 3) + ' Время: ' + Math.min(3, c.time).toFixed(0) + ' / 3 с');
  return lines.join('\n');
}

// ---------- Пошаговый замер по краям ----------
const STEP_TEXT = {
  L: 'ЛЕВЫЙ УГОЛ: наведите на левый угол стены так, чтобы у пола были видны обе стены',
  R: 'ПРАВЫЙ УГОЛ: наведите на правый угол так, чтобы у пола были видны обе стены',
  H: 'ПОТОЛОК: наведите на верх стены, чтобы был виден стык с потолком',
};
const ERR = {nobase: 'Не вижу стык стены с полом', nocorner: 'Угол не найден — нужны обе стены у пола', noheight: 'Стык с потолком не найден', mismatch: 'Правый угол не на одной линии с левым — переснимите'};
function resetWall(){ S.step = 'L'; S.CL = S.CR = null; S.hs = []; S.status = ''; S.fails = 0; S.mode = S.ready ? 'aim' : 'idle'; S.steady = 0; S.pending = null; }
export function reset(){ resetWall(); }
export function tap(){ if (S.cal.ok && S.ready && S.mode !== 'busy' && S.step !== 'done'){ S.mode = 'aim'; S.steady = 0; S.status = ''; } }
export function setActive(on){
  S.active = on;
  if (thumb) thumb.style.display = on && S.last ? 'block' : 'none';
  if (!on) [quad, postL, postR, lineB, lineT, ring].forEach(o => o && (o.visible = false));
  if (on){ load(); if (S.ready && S.mode === 'idle') S.mode = 'aim'; }
}
export const info = () => !S.cal.ok ? 'калибровка' : S.ms ? `нейросеть ${S.ms} мс` : '';
const meter = n => '▮'.repeat(n) + '▯'.repeat(5 - n);
const H = () => S.hs.length >= 5 ? median(S.hs) : null;
export function hint(){
  if (!S.cal.ok) return calibHint();
  if (!S.ready) return S.status || 'Загрузка нейросети…';
  const done = [S.CL ? '✓ лев. угол' : '○ лев. угол', S.CR ? '✓ прав. угол' : '○ прав. угол', H() ? '✓ потолок' : '○ потолок'].join('  ');
  if (S.step === 'done') return done + '\nСтена измерена — нажмите ✓ (↶ — заново)';
  let line = STEP_TEXT[S.step];
  if (S.mode === 'busy') line = 'Анализ кадра… можно двигаться';
  else if (S.mode === 'aim') line += '\nЗамрите ' + meter(Math.min(5, Math.round(S.steady / STEADY_FRAMES * 5)));
  else if (S.status) line = S.status + '\nКоснитесь экрана — снять ещё раз' + (S.step === 'H' ? '' : ', или наведите крестик на угол у пола и нажмите ✓');
  if (S.step === 'H' && S.mode !== 'busy') line += '\n(или наведите крестик на линию потолка над углом и нажмите ✓)';
  return done + '\n' + line;
}
export function canSave(){
  if (!S.cal.ok) return false;
  if (S.step === 'done') return true;
  if (S.mode === 'busy') return false;
  if (S.step === 'L' || S.step === 'R') return !!(S.app && S.app.hit);
  if (S.step === 'H') return S.liveH > 0.5;
  return false;
}
function nextStep(){
  if (!S.CL) S.step = 'L'; else if (!S.CR) S.step = 'R'; else if (!H()) S.step = 'H'; else S.step = 'done';
  S.mode = S.step === 'done' ? 'wait' : 'aim'; S.steady = 0; S.fails = 0;
}
function acceptCorner(K){
  if (S.step === 'R' && S.CL && S.CL.dx != null && K.dx != null){
    const ang = Math.acos(Math.min(1, Math.abs(S.CL.dx*K.dx + S.CL.dz*K.dz)));
    const off = Math.abs((K.x - S.CL.x)*(-S.CL.dz) + (K.z - S.CL.z)*S.CL.dx);
    if (ang > 0.2 || off > 0.12) return false;
  }
  S['C' + S.step] = K; return true;
}

function load(){
  if (S.worker) return;
  S.status = 'Загрузка нейросети…';
  try { S.worker = new Worker(new URL('./segworker.js?v=5', import.meta.url), {type: 'module'}); }
  catch (e){ S.status = 'Браузер не поддерживает фоновые потоки: ' + (e.message || e); return; }
  S.worker.onmessage = ({data: m}) => {
    if (m.type === 'progress') S.status = 'Загрузка нейросети… ' + m.pct + '%';
    else if (m.type === 'ready'){ S.ready = true; S.status = ''; if (S.mode === 'idle') S.mode = 'aim'; }
    else if (m.type === 'result') onResult(m);
    else if (m.type === 'error'){ S.status = 'Ошибка нейросети: ' + m.message; if (S.mode === 'busy') S.mode = 'wait'; }
  };
  S.worker.onerror = e => { S.status = 'Не удалось запустить нейросеть: ' + (e.message || 'ошибка загрузки'); };
  S.worker.postMessage({type: 'load'});
}

// Модуль 8th Wall: сразу после GlTextureRenderer; снимает кадр, когда телефон неподвижен и трекинг NORMAL
export function grabModule(getCam){
  return {
    name: 'wall-grab',
    onStart: ({GLctx}) => { S.gl = GLctx; },
    onRender: () => {
      if (!S.active || !S.cal.ok || S.mode !== 'aim' || !S.ready || S.step === 'done') return;
      const cam = getCam(); if (!cam || !S.prevPos) return;
      cam.updateMatrixWorld();
      const pos = new cam.position.constructor().setFromMatrixPosition(cam.matrixWorld);
      const q = new cam.quaternion.constructor().setFromRotationMatrix(cam.matrixWorld);
      const still = pos.distanceTo(S.prevPos) < 0.004 && q.angleTo(S.prevQ) < 0.006;
      S.prevPos.copy(pos); S.prevQ.copy(q);
      S.steady = still && (!S.trk || S.trk === 'NORMAL') ? S.steady + 1 : 0;
      if (S.steady >= STEADY_FRAMES){ S.steady = 0; capture(cam); }
    },
    listeners: [{event: 'reality.trackingstatus', process: ({detail}) => { S.trk = detail.status; }}],
  };
}
function capture(cam){
  const gl = S.gl; if (!gl) return;
  const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
  const prev = gl.getParameter(gl.FRAMEBUFFER_BINDING);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (!S.buf || S.buf.length !== w*h*4) S.buf = new Uint8Array(w*h*4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, S.buf);
  gl.bindFramebuffer(gl.FRAMEBUFFER, prev);
  const camW = Array.from(cam.matrixWorld.elements);
  const projInv = Array.from(cam.projectionMatrix.clone().invert().elements);
  const portrait = h >= w, Lg = 512;
  const Sh = Math.max(224, Math.round(Lg * Math.min(w, h) / Math.max(w, h) / 32) * 32);
  const W = portrait ? Sh : Lg, Hh = portrait ? Lg : Sh;
  const data = new Float32Array(3*W*Hh), mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  for (let y = 0; y < Hh; y++){
    const sy = h - 1 - Math.min(h - 1, Math.floor((y + 0.5) * h / Hh));
    for (let x = 0; x < W; x++){
      const sx = Math.min(w - 1, Math.floor((x + 0.5) * w / W)), i = (sy*w + sx)*4, o = y*W + x;
      data[o] = (S.buf[i]/255 - mean[0]) / std[0];
      data[W*Hh + o] = (S.buf[i+1]/255 - mean[1]) / std[1];
      data[2*W*Hh + o] = (S.buf[i+2]/255 - mean[2]) / std[2];
    }
  }
  const id = Date.now();
  S.pending = {id, camW, projInv, groundY: S.groundY, step: S.step};
  S.mode = 'busy';
  if (navigator.vibrate) navigator.vibrate(20);
  S.worker.postMessage({type: 'run', id, data, H: Hh, W}, [data.buffer]);
}
function onResult(m){
  const P = S.pending; if (!P || P.id !== m.id) return;
  S.pending = null; S.ms = m.ms;
  if (P.step !== S.step){ S.mode = 'aim'; return; } // пользователь успел сменить шаг
  const [, C, Hl, Wl] = m.dims;
  const A = analyze({logits: m.logits, C, Hl, Wl, camW: P.camW, projInv: P.projInv, groundY: P.groundY});
  S.last = A; drawThumb(A);
  const fail = code => { S.status = ERR[code]; S.mode = 'wait'; S.fails++; };
  if (S.step === 'L' || S.step === 'R'){
    const K = findCorner(A, S.step);
    if (K.error) return fail(K.error);
    if (!acceptCorner(K)) return fail('mismatch');
    S.hs.push(...heightsFrom(A, K.x, K.z, K.dx, K.dz, P.groundY, K.frontCols));
  } else if (S.step === 'H'){
    const dx = S.CR.x - S.CL.x, dz = S.CR.z - S.CL.z, l = Math.hypot(dx, dz);
    const hs = heightsFrom(A, S.CL.x, S.CL.z, dx/l, dz/l, P.groundY, null);
    if (hs.length < 5) return fail('noheight');
    S.hs.push(...hs);
  }
  S.status = ''; nextStep();
}
function drawThumb(A){
  if (!thumb || !A.lab) return;
  const {lab, Wl, Hl} = A; thumb.width = Wl; thumb.height = Hl;
  const c2 = thumb.getContext('2d'), img = c2.createImageData(Wl, Hl);
  for (let p = 0; p < Wl*Hl; p++){
    const k = lab[p];
    const col = k === WALL ? [56, 189, 248] : FLOORISH.has(k) ? [34, 197, 94] : k === CEIL ? [250, 204, 21] : WALLISH.has(k) ? [129, 140, 248] : [50, 50, 50];
    img.data[p*4] = col[0]; img.data[p*4+1] = col[1]; img.data[p*4+2] = col[2]; img.data[p*4+3] = 210;
  }
  c2.putImageData(img, 0, 0); thumb.style.display = S.active ? 'block' : 'none';
}

// Каждый кадр: отрисовка закреплённых краёв (нейросеть не участвует)
export function update(cam){
  const {THREE, setBar, placeLabel, labW, labH, cm} = ctx;
  [quad, postL, postR, lineB, lineT, ring].forEach(o => o.visible = false);
  labW.classList.add('hidden'); labH.classList.add('hidden');
  if (!S.active || !S.cal.ok) return;
  const V = THREE.Vector3, gy = S.groundY, hh = H(), top = gy + (hh || 2.5);
  const e = cam.matrixWorld.elements, o = [e[12], e[13], e[14]];
  const d = new V(0, 0, -1).transformDirection(cam.matrixWorld);
  // прицел на полу для ручного угла
  if ((S.step === 'L' || S.step === 'R') && S.app && S.app.hit && S.mode !== 'busy'){ ring.position.copy(S.app.hit); ring.visible = true; }
  const post = (m, K) => { const a = new V(K.x, gy, K.z), b = new V(K.x, top, K.z); setBar(m, a, b); return a; };
  const A = S.CL && post(postL, S.CL), B = S.CR && post(postR, S.CR);
  // живая высота для ручного шага «потолок»
  S.liveH = null;
  if (S.step === 'H' && S.CL){
    const base = S.CR || S.CL;
    S.liveH = heightOnVertical(base.x, gy, base.z, o, [d.x, d.y, d.z]);
    if (S.liveH > 0.5) placeLabel(labH, new V(base.x, gy + S.liveH/2, base.z), cam, 'В ' + cm(S.liveH));
  }
  if (A && B){
    setBar(lineB, A, B);
    const At = A.clone(), Bt = B.clone(); At.y = Bt.y = top;
    if (hh) setBar(lineT, At, Bt);
    const dir = B.clone().sub(A).setY(0).normalize();
    quad.position.addVectors(A, Bt).multiplyScalar(0.5);
    quad.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(dir, new V(0, 1, 0), new V(-dir.z, 0, dir.x)));
    quad.scale.set(Math.hypot(B.x - A.x, B.z - A.z), top - gy, 1); quad.material.opacity = hh ? 0.3 : 0.12; quad.visible = true;
    placeLabel(labW, new V().addVectors(A, B).multiplyScalar(0.5), cam, 'Ш ' + cm(Math.hypot(B.x - A.x, B.z - A.z)));
    if (hh) placeLabel(labH, new V(B.x, gy + hh/2, B.z), cam, 'В ' + cm(hh));
  }
}
// Кнопка ✓: сохранить готовую стену или поставить текущий край вручную по прицелу
export function save(addMeasurement, toast){
  if (!canSave()) return;
  if (S.step === 'done'){
    const w = Math.hypot(S.CR.x - S.CL.x, S.CR.z - S.CL.z), h = H();
    addMeasurement({type: 'wall', w, h, engine: '8thwall-auto'});
    toast('Стена сохранена: ' + (w*100).toFixed(1) + ' × ' + (h*100).toFixed(1) + ' см');
    resetWall(); return;
  }
  if (S.step === 'L' || S.step === 'R'){
    const p = S.app.hit, other = S.step === 'L' ? S.CR : S.CL;
    let dx = null, dz = null;
    if (other){ dx = (S.step === 'L' ? other.x - p.x : p.x - other.x); dz = (S.step === 'L' ? other.z - p.z : p.z - other.z); const l = Math.hypot(dx, dz); dx /= l; dz /= l; }
    S['C' + S.step] = {x: p.x, z: p.z, dx, dz, manual: true};
    toast((S.step === 'L' ? 'Левый' : 'Правый') + ' угол поставлен вручную');
  } else if (S.step === 'H'){
    for (let i = 0; i < 5; i++) S.hs.push(S.liveH);
    toast('Высота поставлена вручную');
  }
  S.status = ''; nextStep();
}
