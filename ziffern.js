// Liest handgeschriebene Zahlen aus einer Tabellenzelle:
// Zelle → einzelne Zeichen → kleines neuronales Netz (0–9, A, V, W, anderer Buchstabe).
// Kommas/Punkte und Wiederholungszeichen („ “ ″ ||) werden über ihre Form/Lage erkannt.
(function(global){
'use strict';

let NET = null;

async function loadModel(url){
  const j = await (await fetch(url)).json();
  const L = {};
  for(const [name, t] of Object.entries(j.layers)){
    const bin = atob(t.data), q = new Int8Array(bin.length);
    for(let i=0;i<bin.length;i++) q[i] = (bin.charCodeAt(i) << 24) >> 24;
    const f = new Float32Array(q.length);
    for(let i=0;i<q.length;i++) f[i] = q[i] * t.scale;
    L[name] = { shape: t.shape, w: f };
  }
  NET = { classes: j.classes, L };
  return NET;
}

function conv3(inp, C, H, W, wt, bias, O){
  const out = new Float32Array(O*H*W);
  for(let o=0;o<O;o++){
    const b = bias[o];
    for(let y=0;y<H;y++) for(let x=0;x<W;x++){
      let s = b;
      for(let c=0;c<C;c++){
        const wb = ((o*C+c)*9), ib = c*H*W;
        for(let ky=-1;ky<=1;ky++){
          const yy = y+ky; if(yy<0||yy>=H) continue;
          for(let kx=-1;kx<=1;kx++){
            const xx = x+kx; if(xx<0||xx>=W) continue;
            s += inp[ib+yy*W+xx] * wt[wb+(ky+1)*3+(kx+1)];
          }
        }
      }
      out[(o*H+y)*W+x] = s > 0 ? s : 0;
    }
  }
  return out;
}
function pool2(inp, C, H, W){
  const h = H>>1, w = W>>1, out = new Float32Array(C*h*w);
  for(let c=0;c<C;c++) for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    const i = c*H*W + 2*y*W + 2*x;
    out[(c*h+y)*w+x] = Math.max(inp[i], inp[i+1], inp[i+W], inp[i+W+1]);
  }
  return out;
}
function dense(inp, wt, bias, O, relu){
  const N = inp.length, out = new Float32Array(O);
  for(let o=0;o<O;o++){ let s = bias[o]; const b = o*N; for(let i=0;i<N;i++) s += inp[i]*wt[b+i]; out[o] = relu && s < 0 ? 0 : s; }
  return out;
}
// x: Float32Array(784), Tinte = 1
function classify(x){
  const L = NET.L;
  let a = conv3(x, 1, 28, 28, L['c1.weight'].w, L['c1.bias'].w, 16);
  a = conv3(a, 16, 28, 28, L['c2.weight'].w, L['c2.bias'].w, 32); a = pool2(a, 32, 28, 28);
  a = conv3(a, 32, 14, 14, L['c3.weight'].w, L['c3.bias'].w, 48); a = pool2(a, 48, 14, 14);
  a = dense(a, L['f1.weight'].w, L['f1.bias'].w, 64, true);
  const z = dense(a, L['f2.weight'].w, L['f2.bias'].w, NET.classes.length, false);
  const m = Math.max(...z); let s = 0; const p = Array.from(z, v => { const e = Math.exp(v-m); s += e; return e; });
  return p.map(v => v/s);
}

// wie normalize() im Training: zuschneiden, längere Seite 24 px (Flächenmittel), zentrieren
function toInput(val, W, H, box){
  const w = box.x1-box.x0+1, h = box.y1-box.y0+1, s = 24/Math.max(w,h);
  const nw = Math.max(1, Math.round(w*s)), nh = Math.max(1, Math.round(h*s));
  const out = new Float32Array(784);
  const ox = (28-nw)>>1, oy = (28-nh)>>1;
  for(let y=0;y<nh;y++){
    const sy0 = box.y0 + y*h/nh, sy1 = box.y0 + (y+1)*h/nh;
    for(let x=0;x<nw;x++){
      const sx0 = box.x0 + x*w/nw, sx1 = box.x0 + (x+1)*w/nw;
      let acc = 0, area = 0;
      for(let yy=Math.floor(sy0); yy<Math.ceil(sy1); yy++){
        const fy = Math.min(sy1, yy+1) - Math.max(sy0, yy);
        for(let xx=Math.floor(sx0); xx<Math.ceil(sx1); xx++){
          const fx = Math.min(sx1, xx+1) - Math.max(sx0, xx), a = fx*fy;
          acc += val[yy*W+xx]*a; area += a;
        }
      }
      out[(oy+y)*28+ox+x] = area ? acc/area : 0;
    }
  }
  return out;
}

// Zusammenhängende Tintenflächen finden
function components(ink, W, H){
  const lab = new Int32Array(W*H).fill(-1), comps = [], st = [];
  for(let i=0;i<W*H;i++){
    if(!ink[i] || lab[i] >= 0) continue;
    const c = { x0:W, x1:0, y0:H, y1:0, n:0, px:[] };
    lab[i] = comps.length; st.push(i);
    while(st.length){
      const j = st.pop(), x = j%W, y = (j-x)/W;
      c.n++; c.px.push(j);
      if(x<c.x0)c.x0=x; if(x>c.x1)c.x1=x; if(y<c.y0)c.y0=y; if(y>c.y1)c.y1=y;
      for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++){
        const xx=x+dx, yy=y+dy; if(xx<0||yy<0||xx>=W||yy>=H) continue;
        const k = yy*W+xx; if(ink[k] && lab[k] < 0){ lab[k] = comps.length; st.push(k); }
      }
    }
    comps.push(c);
  }
  return comps;
}
const merge = (a, b) => ({ x0:Math.min(a.x0,b.x0), x1:Math.max(a.x1,b.x1), y0:Math.min(a.y0,b.y0), y1:Math.max(a.y1,b.y1), n:a.n+b.n, px:a.px.concat(b.px) });

function boxOf(px, W){
  const b = { x0:1e9, x1:-1, y0:1e9, y1:-1, n:px.length, px };
  for(const j of px){ const x = j%W, y = (j-x)/W; if(x<b.x0)b.x0=x; if(x>b.x1)b.x1=x; if(y<b.y0)b.y0=y; if(y>b.y1)b.y1=y; }
  return b;
}

// Strichdicke wie in den Trainingsdaten (~4 px bei 28×28): dünne Striche verbreitern
function thickness(x){
  let area = 0, per = 0;
  for(let y=0;y<28;y++) for(let i=0;i<28;i++){
    const k = y*28+i; if(x[k] <= 0.3) continue; area++;
    if(i===0||i===27||y===0||y===27||x[k-1]<=0.3||x[k+1]<=0.3||x[k-28]<=0.3||x[k+28]<=0.3) per++;
  }
  return area / Math.max(1, per/2);
}
function thicken(x){
  for(let it=0; it<3 && thickness(x) < 3.3; it++){
    const y = new Float32Array(784);
    for(let r=0;r<28;r++) for(let c=0;c<28;c++){
      let m = x[r*28+c];
      if(r>0) m = Math.max(m, x[(r-1)*28+c]*0.85); if(r<27) m = Math.max(m, x[(r+1)*28+c]*0.85);
      if(c>0) m = Math.max(m, x[r*28+c-1]*0.85); if(c<27) m = Math.max(m, x[r*28+c+1]*0.85);
      y[r*28+c] = m;
    }
    x = y;
  }
  return x;
}

// Ein Zeichen klassifizieren (nur seine eigenen Pixel)
function classifyComp(c, gray, W, H, val){
  for(const j of c.px) val[j] = 1;
  const inp = thicken(toInput(val, W, H, c));
  const p = classify(inp);
  if(global.ZDEBUG) global.ZDEBUG.push({ inp, p });
  for(const j of c.px) val[j] = 0;
  // sehr schmale Striche sind eine 1
  if((c.x1-c.x0+1) < (c.y1-c.y0+1)*0.28){ p.fill(0.01); p[1] = 0.9; }
  return p;
}
const maxP = p => Math.max(...p);

// Zusammengeschriebene Zeichen trennen, wenn die Teile für das Netz deutlich besser "aussehen"
function maybeSplit(c, gray, W, H, val, charH, depth){
  const p = classifyComp(c, gray, W, H, val);
  const w = c.x1-c.x0+1, h = c.y1-c.y0+1;
  if(depth > 2 || w < Math.max(h, charH)*0.75 || w < 10) return [{ c, p }];
  // sicher erkannte Einheit (A, V, W) nicht zerteilen
  const top = p.indexOf(maxP(p));
  if(top >= 10 && top <= 12 && p[top] > 0.6) return [{ c, p }];
  if(top === 0 && p[0] > 0.9 && w < h*1.6) return [{ c, p }];   // breit geschriebene 0
  const col = new Float32Array(w);
  for(const j of c.px) col[(j%W)-c.x0]++;
  // mehrere Schnittkandidaten an den dünnsten Stellen
  const cands = [];
  for(let x=Math.round(w*0.28); x<=Math.round(w*0.72); x++) cands.push([col[x] + 0.3*(col[x-1]+col[x+1]), x]);
  cands.sort((a,b)=>a[0]-b[0]);
  let best = null;
  for(const [, x] of cands.slice(0,3)){
    const cut = c.x0+x, A = [], B = [];
    for(const j of c.px) ((j%W) < cut ? A : B).push(j);
    if(A.length < 8 || B.length < 8) continue;
    const a = boxOf(A, W), b = boxOf(B, W);
    if((a.y1-a.y0) < h*0.45 || (b.y1-b.y0) < h*0.45) continue;
    const pa = classifyComp(a, gray, W, H, val), pb = classifyComp(b, gray, W, H, val);
    const sc = Math.min(maxP(pa), maxP(pb));
    if(!best || sc > best.sc) best = { sc, a, b };
  }
  const single = maxP(p) - Math.max(0, (w/h - 0.85)*1.2);   // breite Gebilde sind selten eine einzelne Ziffer
  if(best && best.sc > single + 0.08)
    return [...maybeSplit(best.a, gray, W, H, val, charH, depth+1), ...maybeSplit(best.b, gray, W, H, val, charH, depth+1)];
  return [{ c, p }];
}

// info: Ergebnis von Tabelle.cleanCell; tableH: typische Ziffernhöhe in der Tabelle (in Zellpixeln) oder 0
function readCell(info, tableH){
  const { ink, gray, W, H } = info;
  let comps = components(ink, W, H).filter(c => c.n >= Math.max(4, W*H*0.0008));
  // Reste der Tabellenlinien am Rand (lang und dünn, berühren den Rand)
  comps = comps.filter(c => {
    const w = c.x1-c.x0+1, h = c.y1-c.y0+1;
    const edgeX = c.x0 <= 1 || c.x1 >= W-2, edgeY = c.y0 <= 1 || c.y1 >= H-2;
    if(edgeX && w < h*0.3 && (h > H*0.5 || c.x0 <= 0 || c.x1 >= W-1)) return false;
    const nearX = c.x1 < W*0.1 || c.x0 > W*0.9;
    if(nearX && w < h*0.18 && h > H*0.55) return false;
    if(edgeY && h < w*0.3 && w > W*0.2) return false;
    return true;
  });
  if(!comps.length) return { text:'', conf:1, chars:0 };
  const heights = comps.map(c => c.y1-c.y0+1).sort((a,b)=>a-b);
  const tallest = heights[heights.length-1];
  const bigH = heights.filter(h => h >= tallest*0.55);
  const charH = bigH[bigH.length>>1];                  // Median der großen Teile
  // Wiederholungszeichen: nur kleine Striche, oben in der Zelle
  // (nur Striche – keine Ziffernformen wie 0, 3, 9, die in kleiner Druckschrift ähnlich klein sind)
  const strokeLike = c => (c.x1-c.x0+1) < (c.y1-c.y0+1)*0.5;
  if(tableH && tallest < tableH*0.62 && comps.length >= 2 && comps.length <= 4 && comps.every(c => (c.y0+c.y1)/2 < H*0.62 && strokeLike(c)))
    return { text:'"', conf:0.8, ditto:true, chars:0 };
  // Textzeile bestimmen
  const bigs = comps.filter(c => (c.y1-c.y0+1) >= charH*0.6);
  const ys0 = bigs.map(c=>c.y0).sort((a,b)=>a-b), ys1 = bigs.map(c=>c.y1).sort((a,b)=>a-b);
  const top = ys0[ys0.length>>1], base = ys1[ys1.length>>1], mid = (top+base)/2;
  // Komma/Punkt: klein, und beginnt erst unterhalb der Mitte der Zeile
  const seps = [], chars = [];
  for(const c of comps){
    const h = c.y1-c.y0+1, w = c.x1-c.x0+1;
    // Komma: kurzer Strich, beginnt deutlich unter der Oberkante und reicht bis zur Grundlinie (oder darunter)
    if(h < charH*0.8 && c.y0 > top + charH*0.3 && c.y1 >= base - charH*0.12 && w < charH*0.38) seps.push(c);
    else if(h >= charH*0.45) chars.push(c);
    else if(h < charH*0.3 && w < charH*0.3 && c.y1 < mid) {}      // Punkt oben/Fleck → ignorieren
    else chars.push(c);
  }
  if(!chars.length) return { text:'', conf:0.3, chars:0 };
  chars.sort((a,b) => a.x0-b.x0);
  // Teile übereinander gehören zu einem Zeichen (z. B. Fähnchen der 5, Querstrich der 7)
  const merged = [];
  for(const c of chars){
    const P = merged[merged.length-1];
    const ov = P ? Math.min(P.x1, c.x1) - Math.max(P.x0, c.x0) : -1;
    const small = P && Math.min(P.y1-P.y0, c.y1-c.y0) < charH*0.7;
    if(P && small && ov > 0.6*Math.min(P.x1-P.x0+1, c.x1-c.x0+1)) merged[merged.length-1] = merge(P, c);
    else merged.push(c);
  }
  const val = new Float32Array(W*H);
  // Nebeneinander liegende Bruchstücke (z. B. eine 0 mit Lücke) zusammenfügen, wenn das Netz sie so besser erkennt
  for(let i=0;i<merged.length-1;i++){
    const A = merged[i], B = merged[i+1], M = merge(A, B);
    const gap = B.x0 - A.x1, mw = M.x1-M.x0+1;
    if(gap > charH*0.15 || mw > charH*0.95) continue;
    const pA = classifyComp(A, gray, W, H, val), pB = classifyComp(B, gray, W, H, val), pM = classifyComp(M, gray, W, H, val);
    const narrow = (A.x1-A.x0+1) < charH*0.45 && (B.x1-B.x0+1) < charH*0.45;
    if(narrow && maxP(pM) > Math.min(maxP(pA), maxP(pB)) + 0.1){ merged.splice(i, 2, M); i--; }
  }
  const res = [];
  merged.forEach(c => res.push(...maybeSplit(c, gray, W, H, val, charH, 0)));
  res.sort((a,b) => a.c.x0-b.c.x0);
  const classes = NET.classes;
  // Einheiten (A, V, W …) am Ende abschneiden
  let end = res.length;
  while(end > 0){
    const p = res[end-1].p, li = p.indexOf(maxP(p));
    if(li >= 10 && p[li] > 0.45) end--; else break;
  }
  const unit = res.slice(end).map(r => classes[r.p.indexOf(maxP(r.p))]).filter(u => u !== '?').join('');
  let out = '', conf = 1;
  for(let i=0;i<end;i++){
    const { c, p } = res[i];
    let bi = 0; for(let k=1;k<10;k++) if(p[k] > p[bi]) bi = k;   // nur Ziffern
    const dsum = p.slice(0,10).reduce((a,b)=>a+b, 0);
    conf = Math.min(conf, p[bi]/dsum * Math.min(1, dsum+0.3));
    out += classes[bi];
    const next = res[i+1];
    if(i < end-1){
      const gapL = c.x1 - (c.x1-c.x0)*0.35, gapR = next.c.x0 + (next.c.x1-next.c.x0)*0.35;
      if(seps.some(s => (s.x0+s.x1)/2 >= gapL && (s.x0+s.x1)/2 <= gapR) && !out.includes(',')) out += ',';
    }
  }
  const items = res.map((r, i) => {
    const p = r.p; let bi = 0; for(let k=1;k<10;k++) if(p[k] > p[bi]) bi = k;
    const dsum = p.slice(0,10).reduce((a,b)=>a+b, 0);
    const next = res[i+1];
    let sep = false;
    if(next){ const gapL = r.c.x1 - (r.c.x1-r.c.x0)*0.35, gapR = next.c.x0 + (next.c.x1-next.c.x0)*0.35;
      sep = seps.some(s => (s.x0+s.x1)/2 >= gapL && (s.x0+s.x1)/2 <= gapR); }
    const w = r.c.x1-r.c.x0+1, h = r.c.y1-r.c.y0+1;
    return { d: String(bi), conf: p[bi]/dsum * Math.min(1, dsum+0.3), letterP: 1-dsum, top: classes[p.indexOf(maxP(p))], sep, stroke: w < h*0.32, h };
  });
  return { text: out, conf, unit, items, chars: res.length, charH,
    dbg: { split: res.map(r=>r.c), seps } };
}

// Ganze Tabelle lesen: infos[r][c] = Ergebnis von Tabelle.cleanCell oder null (leer). Zeile 0 = Überschriften (wird übersprungen).
function readGrid(infos){
  const R = infos.length, C = infos[0] ? infos[0].length : 0;
  // 1. Durchgang: typische Ziffernhöhe
  const hs = [];
  for(let r=1;r<R;r++) for(let c=0;c<C;c++){ const i = infos[r][c]; if(i){ const x = readCell(i, 0); if(x.chars >= 2) hs.push(x.charH); } }
  hs.sort((a,b)=>a-b);
  const tableH = hs[hs.length>>1] || 0;
  const cells = infos.map((row, r) => row.map(i => (i && r > 0) ? readCell(i, tableH) : null));
  const out = cells.map(row => row.map(() => ({ text:'', conf:1 })));
  const units = [];
  for(let c=0;c<C;c++){
    // Einheit in dieser Spalte? (fast jede Zelle endet mit einem Buchstaben)
    let n = 0, withLetter = 0; const seen = {};
    for(let r=1;r<R;r++){ const x = cells[r][c]; if(!x || !x.items || x.items.length < 2) continue; n++;
      const last = x.items[x.items.length-1];
      if(last.letterP > 0.3){ withLetter++; seen[last.top] = (seen[last.top]||0) + 1; } }
    const unitCol = n >= 2 && withLetter >= n*0.35;
    units[c] = unitCol ? (Object.entries(seen).filter(([k]) => k !== '?' && isNaN(k)).sort((a,b)=>b[1]-a[1])[0] || [''])[0] : '';
    for(let r=1;r<R;r++){
      const x = cells[r][c]; if(!x) continue;
      if(x.ditto){ out[r][c] = { text:'"', conf:0.8 }; continue; }
      let items = x.items || [];
      if(unitCol && items.length >= 2 && (items[items.length-1].letterP > 0.04 || !items[items.length-2].sep)) items = items.slice(0, -1);
      else if(items.length >= 2 && items[items.length-1].letterP > 0.6) items = items.slice(0, -1);
      // „||“ = Wiederholungszeichen
      if(items.length === 2 && items.every(it => it.stroke) && !items[0].sep && tableH && items.every(it => it.h < tableH*0.9)){
        out[r][c] = { text:'"', conf:0.7 }; continue;
      }
      let t = '', conf = 1, comma = false;
      items.forEach((it, i) => { t += it.d; conf = Math.min(conf, it.conf); if(it.sep && !comma && i < items.length-1){ t += ','; comma = true; } });
      // führende 0 ohne Komma ist unmöglich → Komma ergänzen (0,72 statt 072)
      if(/^0\d/.test(t)){ t = '0,' + t.slice(1); conf = Math.min(conf, 0.55); }
      out[r][c] = { text: t, conf };
    }
  }
  // Wiederholungszeichen durch den Wert darüber ersetzen
  for(let r=1;r<R;r++) for(let c=0;c<C;c++){
    if(out[r][c].text !== '"') continue;
    const above = out[r-1][c].text;
    const looks11 = cells[r][c] && !cells[r][c].ditto;
    if(looks11 && above && Math.abs(parseFloat(above.replace(',','.')) - 11) <= 3) out[r][c] = { text:'11', conf:0.5 };
    else out[r][c] = { text: above, conf: Math.min(0.75, out[r-1][c].conf), ditto:true };
  }
  return { rows: out, units, tableH };
}

global.Ziffern = { readGrid, loadModel, readCell, classify, toInput, get ready(){ return !!NET; } };
})(window);
