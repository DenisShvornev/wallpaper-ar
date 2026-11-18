// Авто-режим: левый угол -> правый угол -> потолок по краям, найденным нейросетью.
// Кадр снимается только когда телефон неподвижен; сегментация и разбор идут в segworker.js.
// Каждый найденный край закрепляется точкой в мире AR. Калибровка и пол — в calib.js (главный модуль передаёт floorY).
import {median, findCorner, heightsFrom, heightOnVertical, WALL, CEIL, WALLISH, FLOORISH} from './geom.js?v=6';

const STEADY_FRAMES = 10, STILL_POS = 0.006, STILL_ANG = 0.01, NN_LONG = 512;
const S = {active: false, ready: false, worker: null, device: '', mode: 'idle', // idle | aim | busy | wait
  step: 'L', CL: null, CR: null, hs: [], status: '',
  gl: null, buf: null, floorY: null, ms: 0, last: null, trk: null, steady: 0, pending: null, liveH: null};
let ctx = null, quad, postL, postR, lineB, lineT, ring, thumb;
let pPos, pQ, cPos, cQ, vA, vB, vAt, vBt, vDir, vN, vUp, vD, m4;

export function init(c){
  ctx = c; const {THREE, group} = c;
  quad = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({color: 0x38bdf8, transparent: true, opacity: 0.3, side: THREE.DoubleSide, depthTest: false, depthWrite: false}));
  quad.renderOrder = 9; quad.visible = false; group.add(quad);
  const barGeo = new THREE.CylinderGeometry(0.009, 0.009, 1, 8), barMat = new THREE.MeshBasicMaterial({color: 0x22c55e, depthTest: false});
  const bar = () => { const m = new THREE.Mesh(barGeo, barMat); m.renderOrder = 11; m.visible = false; group.add(m); return m; };
  postL = bar(); postR = bar(); lineB = bar(); lineT = bar();
  ring = new THREE.Mesh(new THREE.RingGeometry(0.035, 0.05, 32).rotateX(-Math.PI/2), new THREE.MeshBasicMaterial({color: 0xffffff, depthTest: false, transparent: true, opacity: .9, side: THREE.DoubleSide}));
  ring.renderOrder = 12; ring.visible = false; group.add(ring);
  const V = THREE.Vector3;
  pPos = new V(); cPos = new V(); pQ = new THREE.Quaternion(); cQ = new THREE.Quaternion();
  vA = new V(); vB = new V(); vAt = new V(); vBt = new V(); vDir = new V(); vN = new V(); vUp = new V(0, 1, 0); vD = new V(); m4 = new THREE.Matrix4();
  if (!thumb){
    thumb = document.createElement('canvas');
    thumb.style.cssText = 'position:absolute;right:8px;top:calc(130px + env(safe-area-inset-top));width:64px;border:1px solid #fff6;border-radius:4px;image-rendering:pixelated;pointer-events:none;display:none';
    document.getElementById('hud').appendChild(thumb);
  }
}
// floorY = null — калибровки нет: авто-режим замирает и сбрасывает незавершённую стену.
export function setFloor(y){
  if (y == null && S.floorY != null) resetWall();
  S.floorY = y;
}

const STEP_TEXT = {
  L: 'ЛЕВЫЙ УГОЛ: наведите на левый угол стены так, чтобы у пола были видны обе стены',
  R: 'ПРАВЫЙ УГОЛ: наведите на правый угол так, чтобы у пола были видны обе стены',
  H: 'ПОТОЛОК: наведите на верх стены, чтобы был виден стык с потолком',
};
const ERR = {nobase: 'Не вижу стык стены с полом', nocorner: 'Угол не найден — нужны обе стены у пола', noheight: 'Стык с потолком не найден', mismatch: 'Правый угол не на одной линии с левым — переснимите'};
function resetWall(){ S.step = 'L'; S.CL = S.CR = null; S.hs = []; S.status = S.ready ? '' : S.status; S.mode = S.ready ? 'aim' : 'idle'; S.steady = 0; S.pending = null; }
export function reset(){ resetWall(); }
export function tap(){
  if (!S.ready){ if (!S.worker) load(); return; }
  if (S.floorY != null && S.mode !== 'busy' && S.step !== 'done'){ S.mode = 'aim'; S.steady = 0; S.status = ''; }
}
export function setActive(on){
  S.active = on;
  if (thumb) thumb.style.display = on && S.last ? 'block' : 'none';
  if (!on) [quad, postL, postR, lineB, lineT, ring].forEach(o => o && (o.visible = false));
  if (on){ load(); if (S.ready && S.mode === 'idle') S.mode = 'aim'; }
}
export const preload = () => load();
export const info = () => S.ms ? `нейросеть ${S.device} ${S.ms} мс` : S.device ? 'нейросеть ' + S.device : '';
const meter = n => '▮'.repeat(n) + '▯'.repeat(5 - n);
const H = () => S.hs.length >= 5 ? median(S.hs) : null;
export function hint(){
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
export function canSave(app){
  if (S.floorY == null) return false;
  if (S.step === 'done') return true;
  if (S.mode === 'busy') return false;
  if (S.step === 'L' || S.step === 'R') return !!(app && app.hit);
  if (S.step === 'H') return S.liveH > 0.5;
  return false;
}
function nextStep(){
  if (!S.CL) S.step = 'L'; else if (!S.CR) S.step = 'R'; else if (!H()) S.step = 'H'; else S.step = 'done';
  S.mode = S.step === 'done' ? 'wait' : 'aim'; S.steady = 0;
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
  try { S.worker = new Worker(new URL('./segworker.js?v=6', import.meta.url), {type: 'module'}); }
  catch (e){ S.status = 'Браузер не поддерживает фоновые потоки: ' + (e.message || e); return; }
  S.worker.onmessage = ({data: m}) => {
    if (m.type === 'progress') S.status = 'Загрузка нейросети… ' + m.pct + '%';
    else if (m.type === 'ready'){ S.ready = true; S.device = m.device === 'webgpu' ? 'WebGPU' : 'WASM'; S.status = ''; if (S.mode === 'idle') S.mode = 'aim'; }
    else if (m.type === 'result') onResult(m);
    else if (m.type === 'error'){
      if (m.stage === 'load'){ S.status = 'Ошибка загрузки нейросети: ' + m.message + '\nКоснитесь экрана — повторить'; S.worker.terminate(); S.worker = null; }
      else { S.status = 'Ошибка нейросети: ' + m.message; S.pending = null; if (S.mode === 'busy') S.mode = 'wait'; }
    }
  };
  S.worker.onerror = e => { S.status = 'Не удалось запустить нейросеть: ' + (e.message || 'ошибка загрузки') + '\nКоснитесь экрана — повторить'; S.worker?.terminate(); S.worker = null; };
  S.worker.postMessage({type: 'load'});
}

// Модуль 8th Wall: сразу после GlTextureRenderer — в буфере ещё только кадр камеры, без наших оверлеев.
export function grabModule(getCam){
  return {
    name: 'wall-grab',
    onStart: ({GLctx}) => { S.gl = GLctx; },
    onRender: () => {
      if (!S.active || S.floorY == null || S.mode !== 'aim' || !S.ready || S.step === 'done' || !pPos) return;
      const cam = getCam(); if (!cam) return;
      cam.updateMatrixWorld();
      cam.matrixWorld.decompose(cPos, cQ, vD);
      const still = cPos.distanceTo(pPos) < STILL_POS && cQ.angleTo(pQ) < STILL_ANG;
      pPos.copy(cPos); pQ.copy(cQ);
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
  const portrait = h >= w;
  const Sh = Math.max(224, Math.round(NN_LONG * Math.min(w, h) / Math.max(w, h) / 32) * 32);
  const W = portrait ? Sh : NN_LONG, Hh = portrait ? NN_LONG : Sh;
  // Уменьшение с усреднением 2×2 (меньше шума, чем ближайший пиксель); нормализация — в воркере.
  const rgba = new Uint8Array(W*Hh*4), buf = S.buf, row = w*4;
  for (let y = 0; y < Hh; y++){
    const sy = Math.max(1, h - 1 - Math.min(h - 1, Math.floor((y + 0.5) * h / Hh)));
    for (let x = 0; x < W; x++){
      const sx = Math.min(w - 2, Math.floor((x + 0.5) * w / W)), i = sy*row + sx*4, j = i - row, o = (y*W + x)*4;
      rgba[o]   = (buf[i]   + buf[i+4] + buf[j]   + buf[j+4]) >> 2;
      rgba[o+1] = (buf[i+1] + buf[i+5] + buf[j+1] + buf[j+5]) >> 2;
      rgba[o+2] = (buf[i+2] + buf[i+6] + buf[j+2] + buf[j+6]) >> 2;
    }
  }
  const id = Date.now();
  S.pending = {id, groundY: S.floorY, step: S.step};
  S.mode = 'busy';
  if (navigator.vibrate) navigator.vibrate(20);
  S.worker.postMessage({type: 'run', id, rgba, W, H: Hh, camW, projInv, groundY: S.floorY}, [rgba.buffer]);
}
function onResult(A){
  const P = S.pending; if (!P || P.id !== A.id) return;
  S.pending = null; S.ms = A.ms;
  if (P.step !== S.step || S.floorY == null){ S.mode = 'aim'; return; } // шаг сменился или калибровка сброшена
  S.last = A; drawThumb(A);
  const fail = code => { S.status = ERR[code]; S.mode = 'wait'; };
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
const PAL = new Uint8Array(256*3);
for (let k = 0; k < 256; k++){
  const c = k === WALL ? [56, 189, 248] : FLOORISH.has(k) ? [34, 197, 94] : k === CEIL ? [250, 204, 21] : WALLISH.has(k) ? [129, 140, 248] : [50, 50, 50];
  PAL.set(c, k*3);
}
function drawThumb(A){
  if (!thumb || !A.lab) return;
  const {lab, Wl, Hl} = A; thumb.width = Wl; thumb.height = Hl;
  const c2 = thumb.getContext('2d'), img = c2.createImageData(Wl, Hl), d = img.data;
  for (let p = 0, q = 0; p < Wl*Hl; p++, q += 4){ const k = lab[p]*3; d[q] = PAL[k]; d[q+1] = PAL[k+1]; d[q+2] = PAL[k+2]; d[q+3] = 210; }
  c2.putImageData(img, 0, 0); thumb.style.display = S.active ? 'block' : 'none';
}

// Каждый кадр: отрисовка закреплённых краёв (нейросеть не участвует).
export function update(cam, app){
  const {setBar, placeLabel, labW, labH, cm} = ctx;
  for (const o of [quad, postL, postR, lineB, lineT, ring]) o.visible = false;
  S.liveH = null;
  if (!S.active || S.floorY == null) return;
  const gy = S.floorY, hh = H(), top = gy + (hh || 2.5);
  const e = cam.matrixWorld.elements;
  vD.set(0, 0, -1).transformDirection(cam.matrixWorld);
  if ((S.step === 'L' || S.step === 'R') && app.hit && S.mode !== 'busy'){ ring.position.copy(app.hit); ring.visible = true; }
  const A = S.CL && vA.set(S.CL.x, gy, S.CL.z), B = S.CR && vB.set(S.CR.x, gy, S.CR.z);
  if (A){ vAt.set(A.x, top, A.z); setBar(postL, A, vAt); }
  if (B){ vBt.set(B.x, top, B.z); setBar(postR, B, vBt); }
  if (S.step === 'H' && S.CL){
    const base = S.CR || S.CL;
    S.liveH = heightOnVertical(base.x, gy, base.z, e[12], e[13], e[14], vD.x, vD.y, vD.z);
    if (S.liveH > 0.5) placeLabel(labH, vN.set(base.x, gy + S.liveH/2, base.z), cam, 'В ' + cm(S.liveH));
  }
  if (A && B){
    setBar(lineB, A, B);
    vAt.set(A.x, top, A.z); vBt.set(B.x, top, B.z);
    if (hh) setBar(lineT, vAt, vBt);
    vDir.subVectors(B, A).setY(0).normalize();
    quad.position.addVectors(A, vBt).multiplyScalar(0.5);
    vN.set(-vDir.z, 0, vDir.x);
    quad.quaternion.setFromRotationMatrix(m4.makeBasis(vDir, vUp, vN));
    const w = Math.hypot(B.x - A.x, B.z - A.z);
    quad.scale.set(w, top - gy, 1); quad.material.opacity = hh ? 0.3 : 0.12; quad.visible = true;
    placeLabel(labW, vN.addVectors(A, B).multiplyScalar(0.5), cam, 'Ш ' + cm(w));
    if (hh) placeLabel(labH, vN.set(B.x, gy + hh/2, B.z), cam, 'В ' + cm(hh));
  }
}
// Кнопка ✓: сохранить готовую стену или поставить текущий край вручную по прицелу.
export function save(addMeasurement, toast, app){
  if (!canSave(app)) return;
  if (S.step === 'done'){
    const w = Math.hypot(S.CR.x - S.CL.x, S.CR.z - S.CL.z), h = H();
    addMeasurement({type: 'wall', w, h, engine: '8thwall-auto'});
    toast('Стена сохранена: ' + (w*100).toFixed(1) + ' × ' + (h*100).toFixed(1) + ' см');
    resetWall(); return;
  }
  if (S.step === 'L' || S.step === 'R'){
    const p = app.hit, other = S.step === 'L' ? S.CR : S.CL;
    let dx = null, dz = null;
    if (other){ dx = (S.step === 'L' ? other.x - p.x : p.x - other.x); dz = (S.step === 'L' ? other.z - p.z : p.z - other.z); const l = Math.hypot(dx, dz) || 1; dx /= l; dz /= l; }
    S['C' + S.step] = {x: p.x, z: p.z, dx, dz, manual: true};
    toast((S.step === 'L' ? 'Левый' : 'Правый') + ' угол поставлен вручную');
  } else if (S.step === 'H'){
    for (let i = 0; i < 5; i++) S.hs.push(S.liveH);
    toast('Высота поставлена вручную');
  }
  S.status = ''; nextStep();
}
