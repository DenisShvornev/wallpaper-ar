// Авто-определение стены: SegFormer-B0 (ADE20K) в браузере + геометрия по плоскости пола из SLAM.
// Модель: Xenova/segformer-b0-finetuned-ade-512-512 (4.4 МБ, q8), Transformers.js в фоновом потоке (segworker.js).
// Нейросеть запускается только на «хороших» кадрах: телефон неподвижен, трекинг NORMAL. Результат закрепляется в мире AR.
export const WALL = 0, CEIL = 5;
export const WALLISH = new Set([0, 1, 8, 14, 18, 22, 25, 27]); // wall, building, windowpane, door, curtain, painting, house, mirror
export const FLOORISH = new Set([3, 6, 11, 13, 28]);           // floor, road, sidewalk, earth, rug

// ---------- Чистая геометрия (без three.js — тестируется в node) ----------
const mul4 = (e, x, y, z, w) => [e[0]*x + e[4]*y + e[8]*z + e[12]*w, e[1]*x + e[5]*y + e[9]*z + e[13]*w, e[2]*x + e[6]*y + e[10]*z + e[14]*w, e[3]*x + e[7]*y + e[11]*z + e[15]*w];
function rayDir(camW, projInv, u, v){
  let p = mul4(projInv, u*2 - 1, 1 - v*2, 0.5, 1); p = [p[0]/p[3], p[1]/p[3], p[2]/p[3]];
  const q = mul4(camW, p[0], p[1], p[2], 1);
  const d = [q[0] - camW[12], q[1] - camW[13], q[2] - camW[14]], n = Math.hypot(...d);
  return [d[0]/n, d[1]/n, d[2]/n];
}
function fitLine(P){
  let best = null;
  for (let it = 0; it < 60; it++){
    const a = P[Math.random()*P.length | 0], b = P[Math.random()*P.length | 0];
    const dx = b.x - a.x, dz = b.z - a.z, len = Math.hypot(dx, dz); if (len < 0.15) continue;
    const inl = P.filter(q => Math.abs((q.x - a.x)*dz/len - (q.z - a.z)*dx/len) < 0.04);
    if (!best || inl.length > best.length) best = inl;
  }
  if (!best || best.length < 5) return null;
  let mx = 0, mz = 0; for (const q of best){ mx += q.x; mz += q.z; } mx /= best.length; mz /= best.length;
  let sxx = 0, sxz = 0, szz = 0; for (const q of best){ const x = q.x - mx, z = q.z - mz; sxx += x*x; sxz += x*z; szz += z*z; }
  const ang = 0.5 * Math.atan2(2*sxz, sxx - szz);
  return {mx, mz, dx: Math.cos(ang), dz: Math.sin(ang)};
}
const median = a => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };

// logits: Float32Array [C, Hl, Wl] (batch 1). Возвращает оценку стены в мировых координатах или {error}.
export function estimateWall({logits, C, Hl, Wl, camW, projInv, groundY}){
  const N = Hl * Wl, lab = new Uint8Array(N);
  for (let p = 0; p < N; p++){ let b = -1e9, bi = 0; for (let k = 0; k < C; k++){ const v = logits[k*N + p]; if (v > b){ b = v; bi = k; } } lab[p] = bi; }
  const L = (k, r, c) => logits[k*N + r*Wl + c];
  const base = [], tops = [];
  for (let c = 0; c < Wl; c++){
    // низ стены: снизу вверх через пол до первой не-пол ячейки
    let r = Hl - 1;
    while (r > Hl * 0.55 && !FLOORISH.has(lab[r*Wl + c])) r--;
    if (FLOORISH.has(lab[r*Wl + c])){
      while (r >= 0 && FLOORISH.has(lab[r*Wl + c])) r--;
      if (r >= 1 && WALLISH.has(lab[r*Wl + c]) && WALLISH.has(lab[(r-1)*Wl + c])){
        const k = lab[r*Wl + c], fk = lab[(r+1)*Wl + c];
        const d0 = L(k, r, c) - L(fk, r, c), d1 = L(k, r+1, c) - L(fk, r+1, c);
        const t = d0 - d1 > 1e-6 ? Math.min(1, Math.max(0, d0 / (d0 - d1))) : 0.5;
        base.push({c, v: (r + 0.5 + t) / Hl});
      }
    }
    // верх стены: сверху вниз через потолок
    if (lab[c] === CEIL){
      let r0 = 0; while (r0 < Hl && lab[r0*Wl + c] === CEIL) r0++;
      if (r0 < Hl - 1 && WALLISH.has(lab[r0*Wl + c])){
        const k = lab[r0*Wl + c];
        const dA = L(k, r0-1, c) - L(CEIL, r0-1, c), dB = L(k, r0, c) - L(CEIL, r0, c);
        const t = dB - dA > 1e-6 ? Math.min(1, Math.max(0, -dA / (dB - dA))) : 0.5;
        tops[c] = (r0 - 0.5 + t) / Hl;
      }
    }
  }
  const cam = [camW[12], camW[13], camW[14]];
  const pts = [];
  for (const b of base){
    const d = rayDir(camW, projInv, (b.c + 0.5) / Wl, b.v);
    if (d[1] > -0.02) continue;
    const t = (groundY - cam[1]) / d[1]; if (t <= 0 || t > 10) continue;
    pts.push({c: b.c, x: cam[0] + d[0]*t, z: cam[2] + d[2]*t});
  }
  const out = {lab, Hl, Wl};
  if (pts.length < 8) return {...out, error: 'nobase'};
  // RANSAC: прямая основания стены (отделяет фронтальную стену от боковых)
  let best = null;
  for (let it = 0; it < 120; it++){
    const a = pts[Math.random()*pts.length | 0], b = pts[Math.random()*pts.length | 0];
    const dx = b.x - a.x, dz = b.z - a.z, len = Math.hypot(dx, dz); if (len < 0.2) continue;
    const ux = dx/len, uz = dz/len;
    const inl = pts.filter(q => Math.abs((q.x - a.x)*uz - (q.z - a.z)*ux) < 0.04);
    if (!best || inl.length > best.length) best = inl;
  }
  if (!best || best.length < 8) return {...out, error: 'noline'};
  // самый длинный непрерывный по столбцам участок
  best.sort((p, q) => p.c - q.c);
  let runs = [[best[0]]];
  for (let i = 1; i < best.length; i++){ if (best[i].c - best[i-1].c > 3) runs.push([]); runs[runs.length-1].push(best[i]); }
  const run = runs.reduce((m, r) => r.length > m.length ? r : m);
  if (run.length < 8) return {...out, error: 'noline'};
  // PCA
  let mx = 0, mz = 0; for (const q of run){ mx += q.x; mz += q.z; } mx /= run.length; mz /= run.length;
  let sxx = 0, sxz = 0, szz = 0; for (const q of run){ const x = q.x - mx, z = q.z - mz; sxx += x*x; sxz += x*z; szz += z*z; }
  const ang = 0.5 * Math.atan2(2*sxz, sxx - szz);
  let dx = Math.cos(ang), dz = Math.sin(ang);
  if (dx*camW[0] + dz*camW[2] < 0){ dx = -dx; dz = -dz; } // направление «вправо по экрану»
  const ts = run.map(q => (q.x - mx)*dx + (q.z - mz)*dz);
  let tMin = Math.min(...ts), tMax = Math.max(...ts), leftClosed = false, rightClosed = false;
  // углы: точки основания соседней (боковой) стены по сторонам -> пересечение прямых.
  // Если угол закрыт мебелью, но видна боковая стена, угол всё равно находится.
  const inRun = new Set(run.map(q => q.c));
  const side = (sel, isLeft) => {
    const sp = pts.filter(q => !inRun.has(q.c) && sel(q.c));
    if (sp.length < 5) return null;
    const L2 = fitLine(sp); if (!L2) return null;
    const cr = dx*L2.dz - dz*L2.dx; if (Math.abs(cr) < 0.5) return null; // угол между стенами > 30°
    const t = ((L2.mx - mx)*L2.dz - (L2.mz - mz)*L2.dx) / cr;       // параметр пересечения на основной прямой
    if (isLeft ? (t > tMin + 0.05 || t < tMin - 3) : (t < tMax - 0.05 || t > tMax + 3)) return null;
    return t;
  };
  const c0 = run[0].c, c1 = run[run.length-1].c;
  const tl = side(c => c < c0, true), tr = side(c => c > c1, false);
  if (tl != null){ tMin = tl; leftClosed = true; }
  if (tr != null){ tMax = tr; rightClosed = true; }
  // высота: верхняя граница стены пересекается с вертикальной плоскостью стены
  const nx = -dz, nz = dx, hs = [];
  for (const q of run){
    if (tops[q.c] == null) continue;
    const d = rayDir(camW, projInv, (q.c + 0.5) / Wl, tops[q.c]);
    const den = d[0]*nx + d[2]*nz; if (Math.abs(den) < 1e-4) continue;
    const t = ((mx - cam[0])*nx + (mz - cam[2])*nz) / den; if (t <= 0) continue;
    const h = cam[1] + d[1]*t - groundY;
    if (h > 1.5 && h < 6) hs.push(h);
  }
  return {...out, p0: [mx, groundY, mz], dir: [dx, 0, dz], tMin, tMax, leftClosed, rightClosed,
    h: hs.length >= 3 ? median(hs) : null, nCols: run.length};
}

// Накопление оценок между кадрами (пользователь водит телефоном)
export function mergeWall(M, E, state, a = 0.3, maxMismatch = 3){
  if (!M) return {p0: [...E.p0], dir: [...E.dir], tMin: E.tMin, tMax: E.tMax, leftClosed: E.leftClosed, rightClosed: E.rightClosed, hs: E.h ? [E.h] : [], h: E.h};
  const dot = M.dir[0]*E.dir[0] + M.dir[2]*E.dir[2];
  const ang = Math.acos(Math.min(1, Math.abs(dot)));
  const off = Math.abs((E.p0[0] - M.p0[0])*(-M.dir[2]) + (E.p0[2] - M.p0[2])*M.dir[0]);
  if (ang > 0.21 || off > 0.15){ state.mismatch = (state.mismatch || 0) + 1; return state.mismatch >= maxMismatch ? (state.mismatch = 0, mergeWall(null, E, state)) : M; }
  state.mismatch = 0;
  let {dir, tMin, tMax, leftClosed, rightClosed} = E;
  if (dot < 0){ dir = dir.map(v => -v); [tMin, tMax] = [-tMax, -tMin]; [leftClosed, rightClosed] = [rightClosed, leftClosed]; }
  const eA = [E.p0[0] + dir[0]*tMin, E.p0[2] + dir[2]*tMin], eB = [E.p0[0] + dir[0]*tMax, E.p0[2] + dir[2]*tMax];
  let nd = [M.dir[0]*(1-a) + dir[0]*a, 0, M.dir[2]*(1-a) + dir[2]*a]; const nl = Math.hypot(nd[0], nd[2]); nd = [nd[0]/nl, 0, nd[2]/nl];
  // сдвигаем точку линии к новой оценке по нормали
  const s = (M.p0[0] - E.p0[0])*dir[0] + (M.p0[2] - E.p0[2])*dir[2];
  const proj = [E.p0[0] + dir[0]*s, E.p0[2] + dir[2]*s];
  const p0 = [M.p0[0]*(1-a) + proj[0]*a, E.p0[1], M.p0[2]*(1-a) + proj[1]*a];
  const tA = (eA[0] - p0[0])*nd[0] + (eA[1] - p0[2])*nd[2], tB = (eB[0] - p0[0])*nd[0] + (eB[1] - p0[2])*nd[2];
  const R = {...M, p0, dir: nd, hs: [...M.hs]};
  if (leftClosed){ R.tMin = M.leftClosed ? M.tMin*(1-a) + tA*a : tA; R.leftClosed = true; } else if (!M.leftClosed) R.tMin = Math.min(M.tMin, tA);
  if (rightClosed){ R.tMax = M.rightClosed ? M.tMax*(1-a) + tB*a : tB; R.rightClosed = true; } else if (!M.rightClosed) R.tMax = Math.max(M.tMax, tB);
  if (E.h){ R.hs.push(E.h); if (R.hs.length > 15) R.hs.shift(); }
  R.h = R.hs.length ? median(R.hs) : null;
  return R;
}

// ---------- Браузерная часть ----------
const S = {active: false, ready: false, loading: false, worker: null, mode: 'idle', // idle | aim | busy | done
  gl: null, buf: null, groundY: 0, wall: null, ms: 0, status: '', merge: {}, last: null, shots: 0,
  trk: null, prevPos: null, prevQ: null, steady: 0, pending: null};
const STEADY_FRAMES = 12;
let ctx = null, quad, eB, eT, eL, eR, thumb;

export function init(c){
  ctx = c; const {THREE, group} = c;
  quad = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({color: 0x38bdf8, transparent: true, opacity: 0.3, side: THREE.DoubleSide, depthTest: false, depthWrite: false}));
  quad.renderOrder = 9; quad.visible = false; group.add(quad);
  const bar = () => { const m = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 1, 8), new THREE.MeshBasicMaterial({color: 0x22c55e, depthTest: false})); m.renderOrder = 11; m.visible = false; group.add(m); return m; };
  eB = bar(); eT = bar(); eL = bar(); eR = bar();
  S.prevPos = new THREE.Vector3(); S.prevQ = new THREE.Quaternion();
  if (!thumb){
    thumb = document.createElement('canvas');
    thumb.style.cssText = 'position:absolute;right:8px;top:calc(84px + env(safe-area-inset-top));width:64px;border:1px solid #fff6;border-radius:4px;image-rendering:pixelated;pointer-events:none;display:none';
    document.getElementById('hud').appendChild(thumb);
  }
}
export const setGround = y => { S.groundY = y; };
export const canSave = () => !!(S.wall && S.wall.h && S.wall.tMax - S.wall.tMin > 0.3);
export const info = () => S.ms ? `нейросеть ${S.ms} мс` : '';
const bar = n => '▮'.repeat(n) + '▯'.repeat(5 - n);
export function hint(){
  const M = S.wall;
  if (!S.ready) return S.status || 'Загрузка нейросети…';
  if (S.mode === 'busy') return 'Анализ кадра… можно двигаться';
  if (S.mode === 'aim'){
    if (S.trk && S.trk !== 'NORMAL') return 'Поводите телефоном по полу, пока трекинг не станет NORMAL';
    return (M ? 'Ещё кадр: ' : '') + 'наведите на стену (виден стык с полом) и замрите ' + bar(Math.min(5, Math.round(S.steady / STEADY_FRAMES * 5)));
  }
  if (!M) return (S.status ? S.status + '. ' : '') + 'Коснитесь экрана, чтобы снять заново';
  const miss = [];
  if (!M.h) miss.push('потолок');
  if (!M.leftClosed) miss.push('левый угол');
  if (!M.rightClosed) miss.push('правый угол');
  if (miss.length) return (S.status ? S.status + '. ' : '') + 'Не найдено: ' + miss.join(', ') + '. Наведите туда и коснитесь экрана';
  return 'Стена найдена — нажмите ✓ (↶ — заново)';
}
export function reset(){ S.wall = null; S.merge = {}; S.status = ''; S.shots = 0; S.mode = S.ready ? 'aim' : 'idle'; S.steady = 0; }
export function tap(){ if (S.ready && S.mode !== 'busy'){ S.mode = 'aim'; S.steady = 0; S.status = ''; } }
export function setActive(on){
  S.active = on;
  if (thumb) thumb.style.display = on && S.last ? 'block' : 'none';
  if (!on) [quad, eB, eT, eL, eR].forEach(o => o && (o.visible = false));
  if (on){ load(); if (S.ready && S.mode !== 'busy' && !S.wall){ S.mode = 'aim'; S.steady = 0; } }
}
function load(){
  if (S.worker) return;
  S.status = 'Загрузка нейросети…';
  try {
    S.worker = new Worker(new URL('./segworker.js', import.meta.url), {type: 'module'});
  } catch (e){ S.status = 'Браузер не поддерживает фоновые потоки: ' + (e.message || e); return; }
  S.worker.onmessage = ({data: m}) => {
    if (m.type === 'progress') S.status = 'Загрузка нейросети… ' + m.pct + '%';
    else if (m.type === 'ready'){ S.ready = true; S.status = ''; if (S.mode === 'idle') S.mode = 'aim'; }
    else if (m.type === 'result') onResult(m);
    else if (m.type === 'error'){ S.status = 'Ошибка нейросети: ' + m.message; if (S.mode === 'busy') S.mode = 'done'; }
  };
  S.worker.onerror = e => { S.status = 'Не удалось запустить нейросеть: ' + (e.message || 'ошибка загрузки'); };
  S.worker.postMessage({type: 'load'});
}

// Модуль 8th Wall: ставится сразу после GlTextureRenderer; ждёт неподвижный кадр с хорошим трекингом
export function grabModule(getCam){
  return {
    name: 'wall-grab',
    onStart: ({GLctx}) => { S.gl = GLctx; },
    onRender: () => {
      if (!S.active || S.mode !== 'aim' || !S.ready) return;
      const cam = getCam(); if (!cam || !S.prevPos) return;
      cam.updateMatrixWorld();
      const pos = new cam.position.constructor().setFromMatrixPosition(cam.matrixWorld);
      const q = new cam.quaternion.constructor().setFromRotationMatrix(cam.matrixWorld);
      const still = pos.distanceTo(S.prevPos) < 0.004 && q.angleTo(S.prevQ) < 0.006;
      S.prevPos.copy(pos); S.prevQ.copy(q);
      const trackingOk = !S.trk || S.trk === 'NORMAL';
      S.steady = still && trackingOk ? S.steady + 1 : 0;
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
  const W = portrait ? Sh : Lg, H = portrait ? Lg : Sh;
  const data = new Float32Array(3*W*H), mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  for (let y = 0; y < H; y++){
    const sy = h - 1 - Math.min(h - 1, Math.floor((y + 0.5) * h / H));
    for (let x = 0; x < W; x++){
      const sx = Math.min(w - 1, Math.floor((x + 0.5) * w / W)), i = (sy*w + sx)*4, o = y*W + x;
      data[o] = (S.buf[i]/255 - mean[0]) / std[0];
      data[W*H + o] = (S.buf[i+1]/255 - mean[1]) / std[1];
      data[2*W*H + o] = (S.buf[i+2]/255 - mean[2]) / std[2];
    }
  }
  const id = Date.now();
  S.pending = {id, camW, projInv, groundY: S.groundY};
  S.mode = 'busy';
  if (navigator.vibrate) navigator.vibrate(20);
  S.worker.postMessage({type: 'run', id, data, H, W}, [data.buffer]);
}
function onResult(m){
  const P = S.pending; if (!P || P.id !== m.id) return;
  S.pending = null; S.ms = m.ms;
  const [, C, Hl, Wl] = m.dims;
  const E = estimateWall({logits: m.logits, C, Hl, Wl, camW: P.camW, projInv: P.projInv, groundY: P.groundY});
  S.last = E; drawThumb(E); S.mode = 'done';
  if (E.error){ S.status = E.error === 'nobase' ? 'Не видно стыка стены с полом' : 'Стена не распознана'; return; }
  const before = S.wall;
  S.wall = mergeWall(S.wall, E, S.merge, 1 / (S.shots + 1), 2);
  if (before && S.wall === before) S.status = 'Похоже, это другая стена (↶ — начать заново)';
  else { S.status = ''; S.shots++; }
}
function drawThumb(E){
  if (!thumb || !E.lab) return;
  const {lab, Wl, Hl} = E; thumb.width = Wl; thumb.height = Hl;
  const c2 = thumb.getContext('2d'), img = c2.createImageData(Wl, Hl);
  for (let p = 0; p < Wl*Hl; p++){
    const k = lab[p];
    const col = k === WALL ? [56, 189, 248] : FLOORISH.has(k) ? [34, 197, 94] : k === CEIL ? [250, 204, 21] : WALLISH.has(k) ? [129, 140, 248] : [50, 50, 50];
    img.data[p*4] = col[0]; img.data[p*4+1] = col[1]; img.data[p*4+2] = col[2]; img.data[p*4+3] = 210;
  }
  c2.putImageData(img, 0, 0); thumb.style.display = S.active ? 'block' : 'none';
}

// Каждый кадр: только отрисовка закреплённой стены (без нейросети)
export function update(cam){
  const {THREE, setBar, placeLabel, labW, labH, cm} = ctx, M = S.wall;
  [quad, eB, eT, eL, eR].forEach(o => o.visible = false);
  labW.classList.add('hidden'); labH.classList.add('hidden');
  if (!S.active || !M) return;
  const V = THREE.Vector3, H = M.h || 2.5, gy = M.p0[1];
  const dir = new V(M.dir[0], 0, M.dir[2]), p0 = new V(...M.p0);
  const A = p0.clone().addScaledVector(dir, M.tMin), B = p0.clone().addScaledVector(dir, M.tMax);
  const At = A.clone(), Bt = B.clone(); At.y = Bt.y = gy + H;
  const n = new V(-dir.z, 0, dir.x);
  quad.position.addVectors(A, Bt).multiplyScalar(0.5);
  quad.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(dir, new V(0, 1, 0), n));
  quad.scale.set(M.tMax - M.tMin, H, 1); quad.material.opacity = M.h ? 0.3 : 0.15; quad.visible = true;
  const col = (m, ok) => m.material.color.setHex(ok ? 0x22c55e : 0xf97316);
  col(eB, true); col(eT, !!M.h); col(eL, M.leftClosed); col(eR, M.rightClosed);
  setBar(eB, A, B); setBar(eT, At, Bt); setBar(eL, A, At); setBar(eR, B, Bt);
  placeLabel(labW, new V().addVectors(A, B).multiplyScalar(0.5), cam, 'Ш ' + cm(M.tMax - M.tMin) + (M.leftClosed && M.rightClosed ? '' : ' …'));
  placeLabel(labH, new V(B.x, gy + H/2, B.z), cam, M.h ? 'В ' + cm(M.h) : 'В ?');
}
export function save(addMeasurement, toast){
  const M = S.wall; if (!canSave()) return;
  const w = M.tMax - M.tMin;
  addMeasurement({type: 'wall', w, h: M.h, engine: '8thwall-auto'});
  toast('Стена сохранена: ' + (w*100).toFixed(1) + ' × ' + (M.h*100).toFixed(1) + ' см' + (M.leftClosed && M.rightClosed ? '' : ' (углы не найдены — ширина может быть занижена)'));
  reset();
}
