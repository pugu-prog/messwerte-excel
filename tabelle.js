// Tabellenlinien im Foto finden und die einzelnen Zellen ausschneiden.
// Arbeitet auch bei schräg fotografierten Blättern (jede Linie wird einzeln als Gerade angepasst).
(function(global){
'use strict';

function grayOf(canvas){
  const W = canvas.width, H = canvas.height;
  const p = canvas.getContext('2d').getImageData(0,0,W,H).data;
  const g = new Uint8ClampedArray(W*H);
  for(let i=0;i<W*H;i++) g[i] = (p[i*4]*77 + p[i*4+1]*150 + p[i*4+2]*29) >> 8;
  return { g, W, H };
}

// dunkel = deutlich dunkler als die Umgebung (gleicht Schatten aus)
function binarize(g, W, H, win, C){
  const I = new Float64Array((W+1)*(H+1));
  for(let y=0;y<H;y++){ let s=0; for(let x=0;x<W;x++){ s += g[y*W+x]; I[(y+1)*(W+1)+x+1] = I[y*(W+1)+x+1] + s; } }
  const b = new Uint8Array(W*H), r = win>>1;
  for(let y=0;y<H;y++){
    const y0 = Math.max(0,y-r), y1 = Math.min(H,y+r+1);
    for(let x=0;x<W;x++){
      const x0 = Math.max(0,x-r), x1 = Math.min(W,x+r+1);
      const sum = I[y1*(W+1)+x1] - I[y0*(W+1)+x1] - I[y1*(W+1)+x0] + I[y0*(W+1)+x0];
      if(g[y*W+x] < sum/((x1-x0)*(y1-y0)) - C) b[y*W+x] = 1;
    }
  }
  return b;
}

// Lange waagrechte Striche (horizontal=true) bzw. senkrechte finden → Geradengleichungen
function findLines(b, W, H, horizontal){
  // Achsen tauschen, damit wir immer "waagrecht" denken
  const A = horizontal ? W : H, B = horizontal ? H : W;
  const at = horizontal ? (a,c) => b[c*W+a] : (a,c) => b[a*W+c];
  const thick = 2;                    // Linien quer etwas verdicken → schräge Linien bleiben lange Läufe
  const minRun = Math.max(20, A/28);
  const gapTol = Math.max(2, Math.round(A/400));
  const mask = new Uint8Array(A*B);   // Index c*A+a
  for(let c=0;c<B;c++){
    let start = -1, last = -1;
    const flush = () => { if(start >= 0 && last-start+1 >= minRun) for(let a=start;a<=last;a++) mask[c*A+a] = 1; };
    for(let a=0;a<A;a++){
      let d = 0;
      for(let k=-thick;k<=thick && !d;k++){ const cc = c+k; if(cc>=0 && cc<B && at(a,cc)) d = 1; }
      if(d){ if(start < 0 || a-last > gapTol+1){ flush(); start = a; } last = a; }
    }
    flush();
  }
  // Zusammenhangskomponenten
  const lab = new Int32Array(A*B).fill(-1), comps = [];
  const stack = [];
  for(let i=0;i<A*B;i++){
    if(!mask[i] || lab[i] >= 0) continue;
    const id = comps.length, cols = new Map();
    let minA = A, maxA = 0;
    lab[i] = id; stack.push(i);
    while(stack.length){
      const j = stack.pop(), a = j % A, c = (j - a)/A;
      if(a<minA) minA=a; if(a>maxA) maxA=a;
      const e = cols.get(a); if(e){ e[0]+=c; e[1]++; } else cols.set(a,[c,1]);
      for(let dc=-1;dc<=1;dc++) for(let da=-1;da<=1;da++){
        const aa=a+da, cc=c+dc; if(aa<0||aa>=A||cc<0||cc>=B) continue;
        const k = cc*A+aa; if(mask[k] && lab[k] < 0){ lab[k] = id; stack.push(k); }
      }
    }
    comps.push({ minA, maxA, cols });
  }
  // Gerade c = m*a + q durch die Mittelpunkte
  let segs = comps.filter(s => s.maxA - s.minA >= A*0.08).map(s => {
    let n=0, sa=0, sc=0, saa=0, sac=0;
    s.cols.forEach(([sum,cnt],a) => { const c = sum/cnt; n++; sa+=a; sc+=c; saa+=a*a; sac+=a*c; });
    const den = n*saa - sa*sa, m = den ? (n*sac - sa*sc)/den : 0;
    return { m, q:(sc - m*sa)/n, a0:s.minA, a1:s.maxA };
  }).filter(s => Math.abs(s.m) < 0.15);
  // Teilstücke derselben Linie zusammenfügen
  segs.sort((u,v) => (u.q+u.m*A/2) - (v.q+v.m*A/2));
  const merged = [];
  const tol = Math.max(4, B/150);
  for(const s of segs){
    const L = merged.find(L => Math.abs((L.q+L.m*s.a0) - (s.q+s.m*s.a0)) < tol && Math.abs((L.q+L.m*s.a1) - (s.q+s.m*s.a1)) < tol);
    if(L){
      const w1 = L.a1-L.a0, w2 = s.a1-s.a0;
      L.m = (L.m*w1 + s.m*w2)/(w1+w2); L.q = (L.q*w1 + s.q*w2)/(w1+w2);
      L.a0 = Math.min(L.a0, s.a0); L.a1 = Math.max(L.a1, s.a1);
    } else merged.push({...s});
  }
  return merged;
}

// Schnittpunkt Waagrechte y = m x + q mit Senkrechte x = n y + r
function cross(h, v){
  const x = (v.m*h.q + v.q) / (1 - v.m*h.m);
  return { x, y: h.m*x + h.q };
}

function detectGrid(canvas){
  const { g, W, H } = grayOf(canvas);
  const b = binarize(g, W, H, Math.max(15, Math.round(Math.min(W,H)/40)|1), 12);
  let hs = findLines(b, W, H, true), vs = findLines(b, W, H, false);
  // Nur Linien behalten, die sich mit mehreren anderen kreuzen (= Teil der Tabelle)
  const tolH = W*0.02, tolV = H*0.02;
  const hits = (h, v) => { const p = cross(h, v); return p.x >= h.a0-tolH && p.x <= h.a1+tolH && p.y >= v.a0-tolV && p.y <= v.a1+tolV; };
  for(let it=0; it<4; it++){
    const hs2 = hs.filter(h => vs.filter(v => hits(h,v)).length >= 2);
    const vs2 = vs.filter(v => hs2.filter(h => hits(h,v)).length >= 2);
    if(hs2.length === hs.length && vs2.length === vs.length) break;
    hs = hs2; vs = vs2;
  }
  if(hs.length < 2 || vs.length < 2) return null;
  hs.sort((a,c) => (a.q + a.m*W/2) - (c.q + c.m*W/2));
  vs.sort((a,c) => (a.q + a.m*H/2) - (c.q + c.m*H/2));
  // zu nah beieinander liegende (doppelte) Linien entfernen
  const dedupe = (ls, size) => ls.filter((l,i) => i === 0 || Math.abs((l.q + l.m*size/2) - (ls[i-1].q + ls[i-1].m*size/2)) > size*0.012);
  hs = dedupe(hs, W); vs = dedupe(vs, H);
  return { hs, vs, W, H };
}

const pixelCache = new WeakMap();

// Zelle (Viereck) entzerren und als eigenes Bild ausschneiden
function cellCanvas(src, grid, r, c, outH){
  const { hs, vs } = grid;
  const p00 = cross(hs[r], vs[c]), p01 = cross(hs[r], vs[c+1]);
  const p10 = cross(hs[r+1], vs[c]), p11 = cross(hs[r+1], vs[c+1]);
  const wPx = Math.hypot(p01.x-p00.x, p01.y-p00.y), hPx = Math.hypot(p10.x-p00.x, p10.y-p00.y);
  if(wPx < 8 || hPx < 8) return null;
  const scale = (outH || 64) / hPx;
  const OW = Math.max(16, Math.round(wPx*scale)), OH = Math.max(16, Math.round(hPx*scale));
  let sx = pixelCache.get(src);
  if(!sx){ sx = src.getContext('2d').getImageData(0,0,src.width,src.height); pixelCache.set(src, sx); }
  const sd = sx.data, SW = src.width, SH = src.height;
  const out = document.createElement('canvas'); out.width = OW; out.height = OH;
  const ox = out.getContext('2d'), od = ox.createImageData(OW, OH);
  const insX = 0.035, insY = 0.09;   // Rand weglassen, damit die Linien nicht mitgelesen werden
  for(let y=0;y<OH;y++){
    const v = insY + (1-2*insY)*(y+0.5)/OH;
    for(let x=0;x<OW;x++){
      const u = insX + (1-2*insX)*(x+0.5)/OW;
      const X = (1-u)*(1-v)*p00.x + u*(1-v)*p01.x + (1-u)*v*p10.x + u*v*p11.x;
      const Y = (1-u)*(1-v)*p00.y + u*(1-v)*p01.y + (1-u)*v*p10.y + u*v*p11.y;
      const xi = Math.min(SW-1, Math.max(0, X|0)), yi = Math.min(SH-1, Math.max(0, Y|0));
      const si = (yi*SW+xi)*4, di = (y*OW+x)*4;
      od.data[di] = sd[si]; od.data[di+1] = sd[si+1]; od.data[di+2] = sd[si+2]; od.data[di+3] = 255;
    }
  }
  ox.putImageData(od, 0, 0);
  return out;
}

// Zelle aufräumen: Kontrast normalisieren, Linienreste am Rand entfernen, leer?
function cleanCell(cv){
  const W = cv.width, H = cv.height, x = cv.getContext('2d');
  const d = x.getImageData(0,0,W,H), p = d.data;
  const g = new Float32Array(W*H);
  for(let i=0;i<W*H;i++) g[i] = 0.299*p[i*4] + 0.587*p[i*4+1] + 0.114*p[i*4+2];
  const sorted = Float32Array.from(g).sort();
  const bg = sorted[Math.floor(sorted.length*0.6)];          // Papierfarbe
  const darkest = sorted[Math.floor(sorted.length*0.004)];
  // Schwelle passt sich an: Bleistift (blass) → niedrig, Kuli (dunkel) → höher
  const thr = Math.max(22, Math.min(80, (bg - darkest)*0.42));
  const ink = []; let dark = 0;
  for(let i=0;i<W*H;i++){ const isInk = g[i] < bg - thr; ink.push(isInk); if(isInk) dark++; }
  // Ränder: Zeilen/Spalten, die fast komplett dunkel sind, sind Tabellenlinien
  const wipeRow = y => { for(let xx=0;xx<W;xx++) ink[y*W+xx] = false; };
  const wipeCol = xx => { for(let y=0;y<H;y++) ink[y*W+xx] = false; };
  for(const y of [...Array(Math.ceil(H*0.18)).keys(), ...Array.from({length:Math.ceil(H*0.18)},(_,k)=>H-1-k)]){
    let n=0; for(let xx=0;xx<W;xx++) if(ink[y*W+xx]) n++; if(n > W*0.45) wipeRow(y);
  }
  for(const xx of [...Array(Math.ceil(W*0.06)).keys(), ...Array.from({length:Math.ceil(W*0.06)},(_,k)=>W-1-k)]){
    let n=0; for(let y=0;y<H;y++) if(ink[y*W+xx]) n++; if(n > H*0.45) wipeCol(xx);
  }
  // Graustufenbild mit gestrecktem Kontrast, Linienreste weiß
  let inkCount = 0, minX=W, maxX=0, minY=H, maxY=0;
  for(let i=0;i<W*H;i++){
    let v = Math.max(0, Math.min(255, (g[i] - darkest) / Math.max(30, bg - darkest) * 255));
    if(!ink[i] && g[i] < bg - thr) v = 255;                      // weggewischte Linie
    if(ink[i]){ inkCount++; const xx=i%W, y=(i-xx)/W; if(xx<minX)minX=xx; if(xx>maxX)maxX=xx; if(y<minY)minY=y; if(y>maxY)maxY=y; }
    p[i*4]=p[i*4+1]=p[i*4+2]=v; p[i*4+3]=255;
  }
  x.putImageData(d,0,0);
  const empty = inkCount < W*H*0.006 || (maxX-minX) < H*0.12 && (maxY-minY) < H*0.25;
  return { empty, box: empty ? null : { x0:minX, x1:maxX, y0:minY, y1:maxY }, inkFrac: inkCount/(W*H), ink: Uint8Array.from(ink, v => v ? 1 : 0), gray: g, W, H };
}

// Nur den beschrifteten Bereich (mit etwas Rand) behalten – das Modell erwartet Text, der das Bild füllt
function cropInk(cv, box){
  const pad = Math.round(cv.height*0.12);
  const x0 = Math.max(0, box.x0-pad), y0 = Math.max(0, box.y0-pad);
  const x1 = Math.min(cv.width-1, box.x1+pad), y1 = Math.min(cv.height-1, box.y1+pad);
  let w = x1-x0+1, h = y1-y0+1;
  const out = document.createElement('canvas');
  // nicht schmaler als hoch, sonst wird eine einzelne Ziffer stark verzerrt
  const W = Math.max(w, h), H = h;
  out.width = W; out.height = H;
  const o = out.getContext('2d'); o.fillStyle = '#fff'; o.fillRect(0,0,W,H);
  o.drawImage(cv, x0, y0, w, h, Math.round((W-w)/2), 0, w, h);
  return out;
}

global.Tabelle = { cropInk, detectGrid, cellCanvas, cleanCell, cross };
})(window);
