// Быстрый захват кадра камеры с GPU без остановки рендера.
//
// Было: синхронный readPixels всего экрана (до 20–40 МБ при DPR 3) — GPU-стопор на каждом кадре нейросети.
// Стало (WebGL2): copyTexSubImage2D экрана в текстуру (GPU→GPU) → generateMipmap (честное усреднение) →
// чтение маленького mip-уровня (~0.3–0.6 МБ) в PBO → fenceSync → данные забираются через кадр-два, когда GPU готов.
// Главный поток не ждёт GPU. Состояние WebGL восстанавливается — three.js и 8th Wall ничего не замечают.
// Если что-то из этого недоступно (WebGL1, MSAA-канвас), автоматически используется старый синхронный путь.
export class GpuGrab {
  constructor(gl){
    this.gl = gl; this.gl2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;
    this.ok = this.gl2; this.tex = null; this.fbo = null; this.pbo = null; this.w = 0; this.h = 0; this.job = null; this.checked = false;
  }
  busy(){ return !!this.job; }
  // Начать захват. target — нужный длинный размер. meta уйдёт обратно в колбэк вместе с пикселями.
  start(target, meta, done){
    if (this.job) return false;
    const gl = this.gl, w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    if (!this.ok) return this._sync(w, h, target, meta, done);
    const rFB = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING), unit = gl.getParameter(gl.ACTIVE_TEXTURE);
    const tex0 = gl.getParameter(gl.TEXTURE_BINDING_2D), pbo0 = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING), pack0 = gl.getParameter(gl.PACK_ALIGNMENT);
    try {
      let lv = 0; while (Math.max(w, h) >> (lv + 1) >= target) lv++;
      const mw = Math.max(1, w >> lv), mh = Math.max(1, h >> lv);
      if (this.w !== w || this.h !== h){
        if (this.tex) gl.deleteTexture(this.tex);
        this.tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, this.tex);
        gl.texStorage2D(gl.TEXTURE_2D, Math.floor(Math.log2(Math.max(w, h))) + 1, gl.RGB8, w, h);   // RGB8: копирование допустимо и из канваса без альфы
        this.fbo ??= gl.createFramebuffer(); this.pbo ??= gl.createBuffer();
        this.w = w; this.h = h;
      }
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
      gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, w, h);
      if (lv) gl.generateMipmap(gl.TEXTURE_2D);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.fbo);
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex, lv);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
      gl.bufferData(gl.PIXEL_PACK_BUFFER, mw*mh*4, gl.STREAM_READ);
      gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
      gl.readPixels(0, 0, mw, mh, gl.RGBA, gl.UNSIGNED_BYTE, 0);
      if (!this.checked){ this.checked = true; if (gl.getError() !== gl.NO_ERROR) throw new Error('gpu grab unsupported'); }
      this.job = {sync: gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0), mw, mh, meta, done, t: performance.now()};
      gl.flush();
    } catch (e){
      this.ok = false; this.job = null;   // навсегда переходим на запасной путь
    } finally {
      gl.pixelStorei(gl.PACK_ALIGNMENT, pack0);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo0);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, rFB);
      gl.bindTexture(gl.TEXTURE_2D, tex0); gl.activeTexture(unit);
    }
    if (!this.ok) return this._sync(w, h, target, meta, done);
    return true;
  }
  // Вызывать каждый кадр: если GPU закончил — забрать пиксели.
  poll(){
    const j = this.job; if (!j) return;
    const gl = this.gl, st = gl.clientWaitSync(j.sync, 0, 0);
    if (st !== gl.ALREADY_SIGNALED && st !== gl.CONDITION_SATISFIED){
      if (performance.now() - j.t > 1000){ gl.deleteSync(j.sync); this.job = null; j.done(null); }   // GPU так и не ответил — бросаем кадр
      return;
    }
    gl.deleteSync(j.sync); this.job = null;
    const px = new Uint8Array(j.mw*j.mh*4), pbo0 = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo); gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, px); gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo0);
    j.done(px, j.mw, j.mh, j.meta);
  }
  _sync(w, h, target, meta, done){
    const gl = this.gl, prev = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (!this.buf || this.buf.length !== w*h*4) this.buf = new Uint8Array(w*h*4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, this.buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, prev);
    done(this.buf, w, h, meta);
    return true;
  }
}

// Уменьшить RGBA-кадр (строки снизу вверх, как отдаёт readPixels) до W×H со строками сверху вниз.
export function resample(src, sw, sh, W, H){
  const out = new Uint8Array(W*H*4), fx = sw / W, fy = sh / H;
  const box = fx >= 2 || fy >= 2;   // при сильном уменьшении (запасной путь без мипмапов) усредняем 2×2
  for (let y = 0; y < H; y++){
    const sy = Math.min(sh - 2, sh - 1 - Math.min(sh - 1, Math.floor((y + 0.5) * fy)));   // GL-строка; пара — строка выше
    for (let x = 0; x < W; x++){
      const sx = Math.min(sw - 2, Math.floor((x + 0.5) * fx)), i = (sy*sw + sx)*4, o = (y*W + x)*4;
      if (box){ const j = i + sw*4;
        out[o] = (src[i] + src[i+4] + src[j] + src[j+4]) >> 2; out[o+1] = (src[i+1] + src[i+5] + src[j+1] + src[j+5]) >> 2; out[o+2] = (src[i+2] + src[i+6] + src[j+2] + src[j+6]) >> 2;
      } else { out[o] = src[i]; out[o+1] = src[i+1]; out[o+2] = src[i+2]; }
    }
  }
  return out;
}
