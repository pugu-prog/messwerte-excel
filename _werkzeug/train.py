# Trainiert einen kleinen CNN-Zeichenerkenner: 0-9, A, V, W, "anderer Buchstabe"
# Ausgabe: weights.json (int8-quantisiert) für die Inferenz im Browser
import io, json, sys, glob, random, base64
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

DATA = sys.argv[1]; OUT = sys.argv[2]
CLASSES = list('0123456789') + ['A', 'V', 'W', '?']
OTHER_LETTERS = set('CEFHKMNPRUXY')   # gut von Ziffern unterscheidbar; O,I,L,S,Z,B,G,Q,D,T,J bewusst weggelassen

def normalize(a):
    """a: 2D uint8, Tinte hell. Zuschneiden, längere Seite auf 24 px, im 28x28-Feld zentrieren (wie im Browser)."""
    ys, xs = np.nonzero(a > 40)
    if len(xs) == 0: return np.zeros((28, 28), np.uint8)
    a = a[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = a.shape; s = 24 / max(h, w)
    nw, nh = max(1, round(w*s)), max(1, round(h*s))
    im = Image.fromarray(a).resize((nw, nh), Image.BOX)
    out = np.zeros((28, 28), np.uint8)
    x0, y0 = (28-nw)//2, (28-nh)//2
    out[y0:y0+nh, x0:x0+nw] = np.asarray(im)
    return out

def load(path, transpose, mapper, limit=None):
    t = pq.read_table(path).to_pylist()
    if limit: random.shuffle(t); t = t[:limit]
    X, Y = [], []
    for r in t:
        y = mapper(r['label'])
        if y is None: continue
        a = np.asarray(Image.open(io.BytesIO(r['image']['bytes'])).convert('L'))
        if transpose: a = a.T
        X.append(normalize(a)); Y.append(y)
    return X, Y

def letter_map(l):
    ch = chr(65+l)
    if ch in 'AVW': return CLASSES.index(ch)
    if ch in OTHER_LETTERS: return 13
    return None

def fonts_digits():
    X, Y = [], []
    paths = [p for p in glob.glob('/System/Library/Fonts/**/*.tt[fc]', recursive=True) + glob.glob('/System/Library/Fonts/Supplemental/*.ttf')]
    keep = [p for p in paths if any(k in p for k in ['Arial', 'Helvetica', 'Verdana', 'Times', 'Georgia', 'Courier', 'Bradley', 'Noteworthy', 'Chalkboard', 'Marker', 'Tahoma', 'Trebuchet', 'Avenir', 'Futura', 'Menlo', 'SFNS'])]
    for p in keep:
        for idx in range(3):
            try: f = ImageFont.truetype(p, 64, index=idx)
            except Exception: break
            for ci, ch in enumerate(CLASSES[:13]):
                im = Image.new('L', (120, 120), 0)
                try: ImageDraw.Draw(im).text((20, 10), ch, fill=255, font=f)
                except Exception: continue
                a = np.asarray(im)
                if a.max() < 100: continue
                for _ in range(6): X.append(normalize(a)); Y.append(ci)
    return X, Y

def add_flag(a):
    """1 mit langem Aufstrich (wie in Luxemburg geschrieben)"""
    ys, xs = np.nonzero(a > 100)
    if len(xs) < 10: return None
    im = Image.fromarray(a.copy()); d = ImageDraw.Draw(im)
    ty = ys.min(); tx = int(xs[ys == ty].mean()); h = ys.max()-ys.min()
    L = random.uniform(0.35, 0.85) * h; ang = random.uniform(0.25, 1.05)
    d.line([(tx, ty), (tx - L*np.cos(ang), ty + L*np.sin(ang))], fill=255, width=random.randint(2, 3))
    return normalize(np.asarray(im))

def add_bar(a):
    """7 mit Querstrich"""
    ys, xs = np.nonzero(a > 100)
    if len(xs) < 10: return a
    im = Image.fromarray(a.copy()); d = ImageDraw.Draw(im)
    my = int(ys.min() + (ys.max()-ys.min()) * random.uniform(0.42, 0.62))
    row = xs[np.abs(ys - my) <= 1]
    if len(row) == 0: return a
    cx = int(row.mean()); L = random.randint(4, 7)
    d.line([(cx-L, my), (cx+L, my + random.randint(-1, 1))], fill=255, width=random.randint(2, 3))
    return normalize(np.asarray(im))

def european(X, Y):
    """Alle 7 bekommen einen Querstrich (sonst sind sie von der kontinentalen 1 nicht zu unterscheiden), 1er zusätzlich mit Aufstrich"""
    extra_X, extra_Y = [], []
    for i in range(len(X)):
        if Y[i] == 7: X[i] = add_bar(X[i])
        elif Y[i] == 1 and random.random() < 0.8:
            f = add_flag(X[i])
            if f is not None: extra_X.append(f); extra_Y.append(1)
    return extra_X, extra_Y

def augment(x):
    # x: (B,1,28,28) in [0,1]; zufällige Drehung/Scherung/Skalierung, Strichdicke
    B = x.shape[0]; dev = x.device
    ang = (torch.rand(B, device=dev)-0.5) * 0.45
    sh = (torch.rand(B, device=dev)-0.5) * 0.5
    sc = 0.85 + torch.rand(B, device=dev)*0.3
    tx = (torch.rand(B, device=dev)-0.5)*0.12; ty = (torch.rand(B, device=dev)-0.5)*0.12
    c, s = torch.cos(ang), torch.sin(ang)
    th = torch.stack([torch.stack([c/sc, -s/sc + sh, tx], 1), torch.stack([s/sc, c/sc, ty], 1)], 1)
    g = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, g, align_corners=False)
    r = torch.rand(B, device=dev)
    dil = F.max_pool2d(x, 3, 1, 1); ero = -F.max_pool2d(-x, 3, 1, 1)
    x = torch.where((r < 0.3)[:, None, None, None], dil, torch.where((r > 0.75)[:, None, None, None], ero, x))
    x = x * (0.6 + 0.4*torch.rand(B, 1, 1, 1, device=dev))
    return x.clamp(0, 1)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv2d(1, 16, 3, padding=1); s.c2 = nn.Conv2d(16, 32, 3, padding=1)
        s.c3 = nn.Conv2d(32, 48, 3, padding=1); s.f1 = nn.Linear(48*7*7, 64); s.f2 = nn.Linear(64, len(CLASSES))
        s.drop = nn.Dropout(0.3)
    def forward(s, x):
        x = F.relu(s.c1(x)); x = F.max_pool2d(F.relu(s.c2(x)), 2)
        x = F.max_pool2d(F.relu(s.c3(x)), 2)
        x = s.drop(F.relu(s.f1(x.flatten(1))))
        return s.f2(x)

if __name__ == '__main__':
    random.seed(1); torch.manual_seed(1)
    Xd, Yd = load(f'{DATA}/digits_train.parquet', False, lambda l: l, limit=120000)
    Xl, Yl = load(f'{DATA}/letters_train.parquet', True, letter_map)
    Xf, Yf = fonts_digits()
    Xe, Ye = european(Xd, Yd)
    european(Xf, Yf)
    print('Europa', len(Xe), 'Ziffern', len(Xd), 'Buchstaben', len(Xl), 'Schriften', len(Xf), flush=True)
    X = np.stack(Xd+Xl+Xf+Xe).astype(np.float32)/255; Y = np.array(Yd+Yl+Yf+Ye)
    Xtd, Ytd = load(f'{DATA}/digits_test.parquet', False, lambda l: l, limit=10000)
    Xtl, Ytl = load(f'{DATA}/letters_test.parquet', True, letter_map)
    european(Xtd, Ytd)
    Xt = np.stack(Xtd+Xtl).astype(np.float32)/255; Yt = np.array(Ytd+Ytl)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    X = torch.tensor(X)[:, None].to(dev); Y = torch.tensor(Y).to(dev)
    Xt = torch.tensor(Xt)[:, None].to(dev); Yt = torch.tensor(Yt).to(dev)
    # Klassen ausgleichen (A/V/W sind seltener)
    cnt = torch.bincount(Y, minlength=len(CLASSES)).float()
    wts = (cnt.sum()/cnt/len(CLASSES)).clamp(max=4)
    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    EP = 8; steps = EP * (len(X)//256); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    for ep in range(EP):
        net.train(); perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X)-255, 256):
            idx = perm[i:i+256]
            loss = F.cross_entropy(net(augment(X[idx])), Y[idx], weight=wts)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(Xt[i:i+2000]).argmax(1) for i in range(0, len(Xt), 2000)])
        acc = (pred == Yt).float().mean().item()
        dacc = (pred[:len(Ytd)] == Yt[:len(Ytd)]).float().mean().item()
        print(f'Epoche {ep+1}: Test gesamt {acc:.4f}, Ziffern {dacc:.4f}', flush=True)
    # Export: int8 je Tensor mit Skalierung
    net = net.cpu(); out = {'classes': CLASSES, 'layers': {}}
    for name, p in net.state_dict().items():
        a = p.numpy().astype(np.float32); sc = float(np.abs(a).max()/127 or 1)
        q = np.round(a/sc).astype(np.int8)
        out['layers'][name] = {'shape': list(a.shape), 'scale': sc, 'data': base64.b64encode(q.tobytes()).decode()}
    json.dump(out, open(OUT, 'w'))
    torch.save(net.state_dict(), OUT + '.pt')
    print('gespeichert', OUT)
