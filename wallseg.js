// Режим «Стены»: нейросеть непрерывно размечает кадр, найденные стены закрепляются в 3D и красятся
// своим цветом. Пользователь ведёт камерой по комнате, наводит крестик на стену и нажимает ✓.
// Сегментация и разбор кадра — в segworker.js; модель комнаты — room.js; позу камеры даёт главный модуль.
import {Room} from './room.js?v=10';
import {WALL, CEIL, WALLISH, FLOORISH} from './geom.js?v=10';
import {GpuGrab, resample} from './grab.js?v=10';

// Нагрузка подстраивается под телефон: размер входа сети и пауза между кадрами зависят от измеренного времени.
// Нейросеть занимает не больше ~60% одного ядра, чтобы трекингу 8th Wall хватало процессора.
const STILL_ANG = 0.02, STILL_POS = 0.01, STILL_FRAMES = 3, DUTY = 0.6;
let nnLong = 512;
const S = {active: false, ready: false, worker: null, device: '', busy: false, status: '',
  gl: null, grab: null, floorY: null, ms: 0, lastRun: 0, still: 0, pending: null, sel: null, last: null, engineTag: ''};
const room = new Room();
let ctx = null, group, thumb, meshes = new Map(), selBars = [], m4, vA, vB, vC, vD, vE, vUp, pPos, pQ, cPos, cQ, cS;

export function init(c){
  ctx = c; const {THREE} = c;
  group = new THREE.Group(); group.visible = false; c.group.add(group);
  const geo = new THREE.CylinderGeometry(0.01, 0.01, 1, 8), mat = new THREE.MeshBasicMaterial({color: 0xffffff, depthTest: false});
  for (let i = 0; i < 4; i++){ const m = new THREE.Mesh(geo, mat); m.renderOrder = 12; m.visible = false; group.add(m); selBars.push(m); }
  const V = THREE.Vector3; vA = new V(); vB = new V(); vC = new V(); vD = new V(); vE = new V(); vUp = new V(0, 1, 0); m4 = new THREE.Matrix4();
  pPos = new V(); cPos = new V(); cS = new V(); pQ = new THREE.Quaternion(); cQ = new THREE.Quaternion();
  if (!thumb){
    thumb = document.createElement('canvas');
    thumb.style.cssText = 'position:absolute;right:8px;top:calc(130px + env(safe-area-inset-top));width:56px;border:1px solid #fff6;border-radius:4px;image-rendering:pixelated;pointer-events:none;display:none;opacity:.85';
    document.getElementById('hud').appendChild(thumb);
  }
}
export const setFloor = y => { S.floorY = y; };
export function reset(){ room.clear(); S.sel = null; }
export const hasContent = () => room.walls.length > 0;
export const tap = () => { if (!S.ready && !S.worker) load(); };
export function setActive(on){
  S.active = on; if (group) group.visible = on;
  if (thumb) thumb.style.display = on && S.last ? 'block' : 'none';
  if (on) load();
}
export const preload = () => load();
export const info = () => S.ms ? `нейросеть ${S.device} ${S.ms} мс · ${nnLong}px${S.grab && !S.grab.ok ? ' · медленный захват' : ''}` : S.device ? 'нейросеть ' + S.device : '';
const cmS = m => Math.round(m * 100) + ' см';

export function hint(){
  if (!S.ready) return S.status || 'Загрузка нейросети…';
  if (S.status) return S.status;
  if (!room.visible().length) return 'Медленно ведите камерой по стенам — так, чтобы был виден низ стены у пола';
  if (!S.sel) return 'Наведите крестик на окрашенную стену';
  const z = room.size(S.sel);
  if (!z.h) return 'Поднимите камеру к потолку — нужен стык стены с потолком';
  if (!z.full) return 'Проведите камерой до обоих углов стены — края уточнятся\n(или нажмите ✓, чтобы сохранить как есть)';
  return S.sel.saved ? 'Эта стена уже сохранена' : 'Стена найдена — нажмите ✓';
}
export const canSave = () => !!(S.sel && room.size(S.sel).h);
export function save(addMeasurement, toast){
  if (!canSave()) return;
  const z = room.size(S.sel);
  addMeasurement({type: 'wall', w: z.w, h: z.h, engine: S.engineTag, partial: !z.full});
  toast('Стена сохранена: ' + cmS(z.w) + ' × ' + cmS(z.h) + (z.full ? '' : ' (края не до углов)'));
  S.sel.saved = true;
}

function load(){
  if (S.worker) return;
  S.status = 'Загрузка нейросети…';
  try { S.worker = new Worker(new URL('./segworker.js?v=10', import.meta.url), {type: 'module'}); }
  catch (e){ S.status = 'Браузер не поддерживает фоновые потоки: ' + (e.message || e); return; }
  S.worker.onmessage = ({data: m}) => {
    if (m.type === 'progress') S.status = 'Загрузка нейросети… ' + m.pct + '%';
    else if (m.type === 'ready'){ S.ready = true; S.device = m.device === 'webgpu' ? 'WebGPU' : 'WASM'; S.status = ''; nnLong = m.device === 'webgpu' ? 512 : 384; }
    else if (m.type === 'result') onResult(m);
    else if (m.type === 'error'){
      if (m.stage === 'load'){ S.status = 'Ошибка загрузки нейросети: ' + m.message + '\nКоснитесь экрана — повторить'; S.worker.terminate(); S.worker = null; }
      else { S.busy = false; S.pending = null; }
    }
  };
  S.worker.onerror = e => { S.status = 'Не удалось запустить нейросеть: ' + (e.message || 'ошибка загрузки') + '\nКоснитесь экрана — повторить'; S.worker?.terminate(); S.worker = null; };
  S.worker.postMessage({type: 'load'});
}

// Модуль 8th Wall сразу после GlTextureRenderer: в буфере только кадр камеры.
// getPose() -> {cam, pos} — камера движка и её позиция в МИРОВЫХ координатах приложения.
export function grabModule(getPose, engineTag){
  S.engineTag = engineTag;
  return {
    name: 'wall-grab',
    onStart: ({GLctx}) => { S.gl = GLctx; S.grab = new GpuGrab(GLctx); },
    onRender: () => {
      if (S.grab) S.grab.poll();                     // забрать пиксели прошлого захвата, если GPU готов
      if (!S.active || !S.ready || S.busy || S.floorY == null || !pPos || document.hidden) return;
      const P = getPose(); if (!P) return;
      P.cam.matrixWorld.decompose(cPos, cQ, cS);
      const still = P.pos.distanceTo(pPos) < STILL_POS && cQ.angleTo(pQ) < STILL_ANG;
      pPos.copy(P.pos); pQ.copy(cQ);
      S.still = still ? S.still + 1 : 0;
      const gap = Math.max(200, S.ms * (1 - DUTY) / DUTY);
      if (S.still >= STILL_FRAMES && performance.now() - S.lastRun > gap) capture(P);
    },
  };
}
function capture(P){
  if (!S.grab || S.grab.busy()) return;
  const cam = P.cam;
  const camW = Array.from(cam.matrixWorld.elements); camW[12] = P.pos.x; camW[13] = P.pos.y; camW[14] = P.pos.z; // поза в мире приложения
  const projInv = Array.from(cam.projectionMatrix.clone().invert().elements);
  const meta = {id: Date.now(), camW, projInv, groundY: S.floorY};
  S.busy = true; S.lastRun = performance.now(); S.still = 0; S.pending = meta;
  S.grab.start(nnLong, meta, (px, sw, sh, m) => {
    if (!px || S.pending !== m){ S.busy = false; if (S.pending === m) S.pending = null; return; }
    const portrait = sh >= sw, Sh = Math.max(224, Math.round(nnLong * Math.min(sw, sh) / Math.max(sw, sh) / 32) * 32);
    const W = portrait ? Sh : nnLong, H = portrait ? nnLong : Sh;
    const rgba = resample(px, sw, sh, W, H);
    S.worker.postMessage({type: 'run', id: m.id, rgba, W, H, camW: m.camW, projInv: m.projInv, groundY: m.groundY}, [rgba.buffer]);
  });
}
// Размер входа по скорости: медленно — уменьшаем (384 → 320), быстро — возвращаем 512.
function adapt(ms){
  const next = ms > 900 ? 320 : ms > 450 ? 384 : ms < 180 ? 512 : nnLong;
  if (next !== nnLong){ nnLong = next; }
}
function onResult(A){
  const P = S.pending; S.busy = false;
  if (!P || P.id !== A.id) return;
  S.pending = null; S.ms = A.ms; S.last = A; adapt(A.ms); drawThumb(A);
  if (S.floorY == null || Math.abs(P.groundY - S.floorY) > 0.05) return;   // пол переустановлен во время анализа
  room.ingest(A, P.groundY);
}
const PAL = new Uint8Array(256*3);
for (let k = 0; k < 256; k++) PAL.set(k === WALL ? [56, 189, 248] : FLOORISH.has(k) ? [34, 197, 94] : k === CEIL ? [250, 204, 21] : WALLISH.has(k) ? [129, 140, 248] : [50, 50, 50], k*3);
function drawThumb(A){
  if (!thumb || !A.lab) return;
  const {lab, Wl, Hl} = A; thumb.width = Wl; thumb.height = Hl;
  const c2 = thumb.getContext('2d'), img = c2.createImageData(Wl, Hl), d = img.data;
  for (let p = 0, q = 0; p < Wl*Hl; p++, q += 4){ const k = lab[p]*3; d[q] = PAL[k]; d[q+1] = PAL[k+1]; d[q+2] = PAL[k+2]; d[q+3] = 210; }
  c2.putImageData(img, 0, 0); thumb.style.display = S.active ? 'block' : 'none';
}

function wallMesh(w){
  let m = meshes.get(w.id);
  if (!m){
    const {THREE} = ctx;
    m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({color: w.color, transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthTest: false, depthWrite: false}));
    m.renderOrder = 8; group.add(m); meshes.set(w.id, m);
  }
  return m;
}
// Каждый кадр: стены цветом, выбор стены прицелом, подписи. pos/dir — луч взгляда в мире приложения.
export function update(cam, pos, dir){
  const {setBar, placeLabel, labW, labH} = ctx;
  for (const b of selBars) b.visible = false;
  S.sel = null;
  if (!S.active || S.floorY == null){ for (const m of meshes.values()) m.visible = false; return; }
  const vis = room.visible(), ids = new Set(vis.map(w => w.id)), gy = S.floorY, rh = room.roomHeight();
  for (const [id, m] of meshes){
    if (ids.has(id)) continue;
    m.visible = false;
    if (!room.walls.some(w => w.id === id)){ group.remove(m); m.geometry.dispose(); m.material.dispose(); meshes.delete(id); }
  }
  S.sel = room.pick([pos.x, pos.y, pos.z], [dir.x, dir.y, dir.z], gy);
  for (const w of vis){
    const z = room.size(w), h = z.h || rh || 2.5, m = wallMesh(w);
    vA.set(w.mx + w.dx*z.e0, gy, w.mz + w.dz*z.e0); vB.set(w.mx + w.dx*z.e1, gy, w.mz + w.dz*z.e1);
    m.position.set((vA.x + vB.x)/2, gy + h/2, (vA.z + vB.z)/2);
    vC.set(w.dx, 0, w.dz); vD.set(-w.dz, 0, w.dx);
    m.quaternion.setFromRotationMatrix(m4.makeBasis(vC, vUp, vD));
    m.scale.set(Math.max(0.01, z.w), h, 1);
    const sel = w === S.sel;
    m.material.opacity = sel ? 0.5 : w.saved ? 0.12 : z.h ? 0.32 : 0.18;
    m.visible = true;
    if (sel){
      vD.copy(vA).setY(gy + h); vE.copy(vB).setY(gy + h);
      setBar(selBars[0], vA, vB); setBar(selBars[1], vD, vE);
      if (z.full){ setBar(selBars[2], vA, vD); setBar(selBars[3], vB, vE); }
      placeLabel(labW, vC.set((vA.x + vB.x)/2, gy + 0.12, (vA.z + vB.z)/2), cam, (z.full ? 'Ш ' : 'Ш ≥ ') + cmS(z.w));
      placeLabel(labH, vC.set(vB.x, gy + h/2, vB.z), cam, z.h ? 'В ' + cmS(z.h) : 'В ?');
    }
  }
}
