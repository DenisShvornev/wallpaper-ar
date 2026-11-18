// Чистая геометрия и расчёты без DOM и three.js: используется в главном потоке, в Web Worker и в node-тестах.
export const WALL = 0, CEIL = 5;
export const WALLISH = new Set([0, 1, 8, 14, 18, 22, 25, 27]); // wall, building, windowpane, door, curtain, painting, house, mirror
export const FLOORISH = new Set([3, 6, 11, 13, 28]);           // floor, road, sidewalk, earth, rug

export const median = a => { if (!a.length) return null; const s = Float64Array.from(a).sort(); return s[s.length >> 1]; };

const mul4 = (e, x, y, z, w) => [e[0]*x + e[4]*y + e[8]*z + e[12]*w, e[1]*x + e[5]*y + e[9]*z + e[13]*w, e[2]*x + e[6]*y + e[10]*z + e[14]*w, e[3]*x + e[7]*y + e[11]*z + e[15]*w];
export function rayDir(camW, projInv, u, v){
  let p = mul4(projInv, u*2 - 1, 1 - v*2, 0.5, 1); p = [p[0]/p[3], p[1]/p[3], p[2]/p[3]];
  const q = mul4(camW, p[0], p[1], p[2], 1);
  const d = [q[0] - camW[12], q[1] - camW[13], q[2] - camW[14]], n = Math.hypot(d[0], d[1], d[2]);
  return [d[0]/n, d[1]/n, d[2]/n];
}

// Уровень пола по набору высот хитов: самый НИЖНИЙ плотный кластер (стол/диван выше пола и не мешают).
export function floorCluster(ys, tol = 0.025, minN = 12){
  const n = ys.length; if (!n) return null;
  const s = Float64Array.from(ys).sort(), cnt = new Int32Array(n);
  let j = 0, k = 0, best = 0;
  for (let i = 0; i < n; i++){
    while (s[j] < s[i] - tol) j++;
    if (k < i) k = i;
    while (k < n && s[k] <= s[i] + tol) k++;
    cnt[i] = k - j; if (cnt[i] > best) best = cnt[i];
  }
  const need = Math.max(Math.min(minN, best), Math.ceil(best * 0.5));
  let i = 0; while (cnt[i] < need) i++;
  // уточняем центр: медиана окна, затем медиана инлаеров вокруг неё
  let win = []; for (let q = 0; q < n; q++) if (Math.abs(s[q] - s[i]) <= tol) win.push(s[q]);
  const c0 = win[win.length >> 1];
  win = []; for (let q = 0; q < n; q++) if (Math.abs(s[q] - c0) <= tol) win.push(s[q]);
  return {y: win[win.length >> 1], n: win.length};
}

function ransacLine(P, iters = 120, tol = 0.04){
  let best = null, bestN = 0;
  for (let it = 0; it < iters; it++){
    const a = P[Math.random()*P.length | 0], b = P[Math.random()*P.length | 0];
    const dx = b.x - a.x, dz = b.z - a.z, len = Math.hypot(dx, dz); if (len < 0.15) continue;
    const nx = dz/len, nz = dx/len; let c = 0;
    for (const q of P) if (Math.abs((q.x - a.x)*nx - (q.z - a.z)*nz) < tol) c++;
    if (c > bestN){ bestN = c; best = [a, nx, nz]; }
  }
  if (bestN < 6) return null;
  const [a, nx, nz] = best, inl = P.filter(q => Math.abs((q.x - a.x)*nx - (q.z - a.z)*nz) < tol);
  let mx = 0, mz = 0; for (const q of inl){ mx += q.x; mz += q.z; } mx /= inl.length; mz /= inl.length;
  let sxx = 0, sxz = 0, szz = 0; for (const q of inl){ const x = q.x - mx, z = q.z - mz; sxx += x*x; sxz += x*z; szz += z*z; }
  const ang = 0.5 * Math.atan2(2*sxz, sxx - szz);
  const mc = inl.reduce((s, q) => s + q.c, 0) / inl.length;
  return {mx, mz, dx: Math.cos(ang), dz: Math.sin(ang), pts: inl, mc};
}

// Разбор кадра сегментации: точки стыка «стена–пол» (в мире) и лучи на стык «стена–потолок» по столбцам.
export function analyze({logits, C, Hl, Wl, camW, projInv, groundY}){
  const N = Hl * Wl, lab = new Uint8Array(N), best = new Float32Array(N).fill(-Infinity);
  for (let k = 0; k < C; k++){          // argmax по классам: последовательный проход по памяти
    const off = k*N;
    for (let p = 0; p < N; p++){ const v = logits[off + p]; if (v > best[p]){ best[p] = v; lab[p] = k; } }
  }
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
  const near = l => { let m = Infinity; for (const q of l.pts) m = Math.min(m, Math.hypot(q.x - x, q.z - z)); return m; };
  if (near(l1) > 1.2 || near(l2) > 1.2) return {error: 'nocorner'}; // угол далеко от видимых стыков — ненадёжно
  const [left, right] = l1.mc < l2.mc ? [l1, l2] : [l2, l1];
  const front = which === 'L' ? right : left;
  let dx = front.dx, dz = front.dz;
  if (dx*A.camW[0] + dz*A.camW[2] < 0){ dx = -dx; dz = -dz; }      // направление вдоль стены «вправо»
  return {x, z, dx, dz, frontCols: new Set(front.pts.map(q => q.c))};
}

// Высоты: лучи на стык с потолком пересекаются с вертикальной плоскостью стены.
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

// Высота над точкой (bx, by, bz): ближайшая точка между лучом взгляда (o, d) и вертикалью через точку.
export function heightOnVertical(bx, by, bz, ox, oy, oz, dx, dy, dz){
  const w0x = ox - bx, w0y = oy - by, w0z = oz - bz;
  const a = dx*dx + dy*dy + dz*dz, b = dy;
  const dd = dx*w0x + dy*w0y + dz*w0z, e = w0y, den = a - b*b;
  if (den < 1e-6) return null;
  if ((b*e - dd) / den <= 0) return null;
  return (a*e - b*dd) / den;
}

// Расчёт рулонов. Все длины в метрах, res — в процентах, k — множитель для оценки погрешности.
export function calcRolls(walls, {rollW, rollL, rap = 0, trim = 0, open = 0, res = 0}, k = 1){
  if (!walls.length || rollW <= 0 || rollL <= 0) return null;
  let strips = 0, maxH = 0, sumW = 0;
  for (const w of walls){ const W = w.w * k, H = w.h * k; strips += Math.ceil(W / rollW - 1e-9); maxH = Math.max(maxH, H); sumW += W; }
  let stripLen = maxH + trim; if (rap > 0) stripLen = Math.ceil(stripLen / rap - 1e-9) * rap;
  const spr = Math.floor(rollL / stripLen + 1e-9);
  if (spr < 1) return {error: 'Полоса длиннее рулона'};
  const need = Math.max(0, strips - Math.floor(open / (rollW * maxH)));
  return {strips: need, spr, stripLen, rolls: Math.ceil(need * (1 + res/100) / spr - 1e-9), sumW, maxH};
}
