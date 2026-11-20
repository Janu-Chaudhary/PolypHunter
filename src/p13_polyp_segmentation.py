# %% [markdown]
# # P13 — Polyp Segmentation in Colonoscopy: U-Net and Its Variants
#
# **Task.** Pixel-level segmentation of polyps in colonoscopy frames (Kvasir-SEG, 1000 images + masks).
#
# **Models compared (each step adds one idea):**
# 1. `unet_scratch` — U-Net baseline trained from scratch (Ronneberger et al., 2015)
# 2. `unet_resnet34` — U-Net with an ImageNet-pretrained ResNet-34 encoder (transfer learning)
# 3. `attention_unet` — Attention U-Net (Oktay et al., 2018): attention gates on the skip connections, from scratch
# 4. `segformer_mitb2` — transformer encoder (MiT-B2, SegFormer), ImageNet-pretrained
# 5. `unet_mobilenetv2` — lightweight U-Net (MobileNetV2 encoder) as a real-time reference point
#
# **Extras.** (A) test-time augmentation, (B) lightweight model, (C) cross-dataset test on CVC-ClinicDB,
# (D) attention-gate maps, (E) threshold tuning on the validation set.
#
# **Metrics.** Dice and IoU (plus precision/recall) on a held-out test set.
#
# **Analyses.** (a) accuracy vs inference speed (FPS, batch 1) for real-time use;
# (b) failure analysis on small and flat (low-contrast) polyps.
#
# References: [1] Ronneberger et al., MICCAI 2015, arXiv:1505.04597 · [2] Jha et al., MMM 2020, arXiv:1911.07069

# %%
# ---------------- configuration ----------------
SMOKE = False          # True = tiny fast run to check the pipeline end to end
IMG_SIZE = 352
BATCH_SIZE = 8
SEED = 42
N_TEST, N_VAL = 120, 80   # 1000 images -> 800 train / 80 val / 120 test (test size follows [2])
EPOCHS = {"unet_scratch": 60, "attention_unet": 60, "unet_resnet34": 40, "segformer_mitb2": 40, "unet_mobilenetv2": 40}
LR = {"unet_scratch": 3e-4, "attention_unet": 3e-4, "unet_resnet34": 2e-4, "segformer_mitb2": 1e-4, "unet_mobilenetv2": 3e-4}
if SMOKE:
    EPOCHS = {k: 1 for k in EPOCHS}

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=False)

# %%
import os, glob, json, time, math, random, zipfile, urllib.request
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("outputs")
os.makedirs(f"{OUT}/figures", exist_ok=True); os.makedirs(f"{OUT}/weights", exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| smp", smp.__version__, "| device", DEVICE,
      torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all()

# %% [markdown]
# ## 1. Data: locate or download Kvasir-SEG

# %%
def find_kvasir(root):
    """Return (images_dir, masks_dir) if a Kvasir-SEG layout is found under root."""
    for img_dir in glob.glob(f"{root}/**/images", recursive=True):
        mask_dir = os.path.join(os.path.dirname(img_dir), "masks")
        if os.path.isdir(mask_dir) and len(os.listdir(img_dir)) >= 900:
            return img_dir, mask_dir
    return None

found = find_kvasir("/kaggle/input") if os.path.isdir("/kaggle/input") else None
if found is None:
    found = find_kvasir("data")
if found is None:
    os.makedirs("data", exist_ok=True)
    url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
    print("downloading", url)
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, "data/kvasir-seg.zip"); break
        except Exception as e:
            print("download failed:", e); time.sleep(10)
    zipfile.ZipFile("data/kvasir-seg.zip").extractall("data")
    found = find_kvasir("data")
IMG_DIR, MASK_DIR = found
names = sorted(os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png")))
print(len(names), "images in", IMG_DIR)

# Kvasir-Sessile (Jha et al. 2021): 196 Kvasir-SEG polyps labelled sessile/flat and <=10 mm -> real "flat" label
SESSILE = {os.path.splitext(os.path.basename(f))[0]
           for f in glob.glob("/kaggle/input/**/sessile-main-Kvasir-SEG/images/*", recursive=True)}
print(len(SESSILE), "sessile (flat) polyp ids found",
      f"({len(SESSILE & set(names))} overlap with Kvasir-SEG)")

def read_pair(name):
    ip = glob.glob(os.path.join(IMG_DIR, name + ".*"))[0]
    mp = glob.glob(os.path.join(MASK_DIR, name + ".*"))[0]
    img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    return img, (m > 127).astype(np.uint8)

# Fixed, reproducible split
rng = np.random.RandomState(SEED)
perm = rng.permutation(names).tolist()
test_names, val_names, train_names = perm[:N_TEST], perm[N_TEST:N_TEST + N_VAL], perm[N_TEST + N_VAL:]
if SMOKE:
    train_names, val_names, test_names = train_names[:32], val_names[:16], test_names[:16]
json.dump({"train": train_names, "val": val_names, "test": test_names}, open(f"{OUT}/split.json", "w"), indent=1)
print(f"train {len(train_names)} | val {len(val_names)} | test {len(test_names)}")

# Preload everything resized into RAM (small dataset). Keep original-resolution mask area for analysis.
def load_set(ns):
    X, Y, area = [], [], []
    for n in ns:
        img, m = read_pair(n)
        area.append(m.mean())
        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
        Y.append(cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST))
    return np.stack(X), np.stack(Y), np.array(area)
Xtr, Ytr, _ = load_set(train_names)
Xva, Yva, _ = load_set(val_names)
Xte, Yte, area_te = load_set(test_names)

# (C) CVC-ClinicDB (612 frames, different hospital/scope) for a cross-dataset test - no retraining
def load_cvc():
    for orig in glob.glob("/kaggle/input/**/Original", recursive=True):
        gt = os.path.join(os.path.dirname(orig), "Ground Truth")
        if not os.path.isdir(gt): continue
        X, Y = [], []
        for f in sorted(os.listdir(orig)):
            g = glob.glob(os.path.join(gt, os.path.splitext(f)[0] + ".*"))
            img = cv2.imread(os.path.join(orig, f))
            if not g or img is None: continue
            m = cv2.imread(g[0], cv2.IMREAD_GRAYSCALE)
            X.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
            Y.append(cv2.resize((m > 127).astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST))
        if len(X) > 100:
            print("CVC-ClinicDB:", len(X), "images from", orig)
            return np.stack(X), np.stack(Y)
    print("CVC-ClinicDB not found - cross-dataset test skipped"); return None, None
Xcvc, Ycvc = load_cvc()
if SMOKE and Xcvc is not None: Xcvc, Ycvc = Xcvc[:16], Ycvc[:16]

# %% [markdown]
# ## 2. Augmentation and data loaders

# %%
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def augment(img, m):
    if random.random() < 0.5: img, m = img[:, ::-1], m[:, ::-1]
    if random.random() < 0.5: img, m = img[::-1], m[::-1]
    k = random.randint(0, 3); img, m = np.rot90(img, k), np.rot90(m, k)
    img, m = np.ascontiguousarray(img), np.ascontiguousarray(m)
    if random.random() < 0.7:  # scale / rotate / shift
        ang, sc = random.uniform(-25, 25), random.uniform(0.85, 1.15)
        M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), ang, sc)
        M[:, 2] += np.random.uniform(-0.06, 0.06, 2) * IMG_SIZE
        img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = cv2.warpAffine(m, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    if random.random() < 0.6:  # brightness / contrast
        a, b = random.uniform(0.8, 1.2), random.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return img, m

class SegDS(torch.utils.data.Dataset):
    def __init__(self, X, Y, train): self.X, self.Y, self.train = X, Y, train
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        img, m = self.X[i], self.Y[i]
        if self.train: img, m = augment(img, m)
        img = (img.astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(img.transpose(2, 0, 1)), torch.from_numpy(m[None].astype(np.float32))

def loader(X, Y, train):
    return torch.utils.data.DataLoader(SegDS(X, Y, train), batch_size=BATCH_SIZE, shuffle=train,
                                       num_workers=2, drop_last=train, pin_memory=True)
dl_tr, dl_va, dl_te = loader(Xtr, Ytr, True), loader(Xva, Yva, False), loader(Xte, Yte, False)
dl_cvc = loader(Xcvc, Ycvc, False) if Xcvc is not None else None

# %% [markdown]
# ## 3. Models
# U-Net and Attention U-Net are written out here (from scratch, same width: 64→1024 channels).
# BatchNorm and padded convs are modern additions to the original U-Net so output size = input size.

# %%
class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class AttentionGate(nn.Module):
    """Oktay et al. 2018: gating signal g (decoder) re-weights skip features x (encoder)."""
    def __init__(self, cg, cx, cint):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(cg, cint, 1, bias=False), nn.BatchNorm2d(cint))
        self.wx = nn.Sequential(nn.Conv2d(cx, cint, 1, bias=False), nn.BatchNorm2d(cint))
        self.psi = nn.Sequential(nn.Conv2d(cint, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
    def forward(self, g, x):
        a = self.psi(F.relu(self.wg(g) + self.wx(x)))
        return x * a

class UNet(nn.Module):
    def __init__(self, ch=(64, 128, 256, 512, 1024), attention=False):
        super().__init__()
        self.attention = attention
        self.enc = nn.ModuleList([DoubleConv(3, ch[0])] + [DoubleConv(ch[i], ch[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(ch[i] * 2, ch[i]) for i in reversed(range(4))])
        if attention:
            self.gates = nn.ModuleList([AttentionGate(ch[i], ch[i], ch[i] // 2) for i in reversed(range(4))])
        self.head = nn.Conv2d(ch[0], 1, 1)
    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x); s = skips.pop()
            if self.attention: s = self.gates[j](x, s)
            x = dec(torch.cat([s, x], 1))
        return self.head(x)

class Resize(nn.Module):
    """Makes sure logits match the input size (some heads predict at 1/4 resolution)."""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        y = self.m(x)
        return y if y.shape[-2:] == x.shape[-2:] else F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

def build(name):
    if name == "unet_scratch": return UNet()
    if name == "attention_unet": return UNet(attention=True)
    if name == "unet_resnet34": return smp.Unet("resnet34", encoder_weights="imagenet", classes=1)
    if name == "unet_mobilenetv2": return smp.Unet("mobilenet_v2", encoder_weights="imagenet", classes=1)
    if name == "segformer_mitb2":
        if hasattr(smp, "Segformer"):
            return Resize(smp.Segformer("mit_b2", encoder_weights="imagenet", classes=1))
        return Resize(smp.FPN("mit_b2", encoder_weights="imagenet", classes=1))
    raise ValueError(name)

MODELS = ["unet_scratch", "unet_resnet34", "unet_mobilenetv2", "attention_unet", "segformer_mitb2"]
for n in MODELS:
    print(f"{n:18s} {sum(p.numel() for p in build(n).parameters()) / 1e6:6.1f} M params")

# %% [markdown]
# ## 4. Loss, metrics, training

# %%
def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (den + eps)).mean()

def loss_fn(logits, y):
    return F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)

def per_image_metrics(pred, y, eps=1e-7):
    """pred, y: binary tensors (B,1,H,W). Returns dict of per-image arrays."""
    pred, y = pred.float(), y.float()
    tp = (pred * y).sum((1, 2, 3)); fp = (pred * (1 - y)).sum((1, 2, 3)); fn = ((1 - pred) * y).sum((1, 2, 3))
    return {"dice": ((2 * tp + eps) / (2 * tp + fp + fn + eps)).cpu().numpy(),
            "iou": ((tp + eps) / (tp + fp + fn + eps)).cpu().numpy(),
            "precision": ((tp + eps) / (tp + fp + eps)).cpu().numpy(),
            "recall": ((tp + eps) / (tp + fn + eps)).cpu().numpy()}

FLIPS = [[], [3], [2], [2, 3]]   # identity, h-flip, v-flip, both  (A: test-time augmentation)

@torch.no_grad()
def predict(model, dl, tta=False):
    model.eval(); probs = []
    for x, _ in dl:
        x = x.to(DEVICE)
        with torch.autocast("cuda", enabled=DEVICE == "cuda"):
            if not tta:
                p = torch.sigmoid(model(x).float())
            else:
                p = sum(torch.sigmoid(model(x.flip(f) if f else x).float()).flip(f) if f
                        else torch.sigmoid(model(x).float()) for f in FLIPS) / len(FLIPS)
        probs.append(p.cpu())
    return torch.cat(probs)

def evaluate(probs, Y, thr=0.5):
    m = per_image_metrics(probs > thr, torch.from_numpy(Y[:, None]))
    return m, {k: float(v.mean()) for k, v in m.items()}

def train(name):
    seed_all()
    model = build(name).to(DEVICE)
    ep = EPOCHS[name]
    opt = torch.optim.AdamW(model.parameters(), lr=LR[name], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR[name], total_steps=ep * len(dl_tr), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    hist, best, t0 = [], -1, time.time()
    for e in range(ep):
        model.train(); tl = 0
        for x, y in dl_tr:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(x)
        _, vm = evaluate(predict(model, dl_va), Yva)
        hist.append({"epoch": e + 1, "train_loss": tl / len(dl_tr.dataset), "val_dice": vm["dice"], "val_iou": vm["iou"]})
        if vm["dice"] > best:
            best = vm["dice"]; torch.save(model.state_dict(), f"{OUT}/weights/{name}.pt")
        print(f"[{name}] ep {e + 1:3d}/{ep} loss {hist[-1]['train_loss']:.4f} val dice {vm['dice']:.4f} iou {vm['iou']:.4f} (best {best:.4f})")
    model.load_state_dict(torch.load(f"{OUT}/weights/{name}.pt", map_location=DEVICE))
    print(f"[{name}] done in {(time.time() - t0) / 60:.1f} min")
    return model, pd.DataFrame(hist), (time.time() - t0) / 60

# %% [markdown]
# ## 5. Train all models

# %%
trained, histories, train_minutes = {}, {}, {}
for n in MODELS:
    trained[n], histories[n], train_minutes[n] = train(n)
    histories[n].to_csv(f"{OUT}/history_{n}.csv", index=False)
    torch.cuda.empty_cache()

# %% [markdown]
# ## 6. Test-set results

# %%
THRESHOLDS = np.round(np.arange(0.2, 0.81, 0.05), 2)
test_probs, per_img, rows = {}, {}, []
for n in MODELS:
    test_probs[n] = predict(trained[n], dl_te)
    per_img[n], mean = evaluate(test_probs[n], Yte)
    row = {"model": n, **{k: round(v, 4) for k, v in mean.items()}}
    # (E) threshold chosen on the validation set only, then applied to test
    vp = predict(trained[n], dl_va)
    thr = float(max(THRESHOLDS, key=lambda t: evaluate(vp, Yva, t)[1]["dice"]))
    _, mt = evaluate(test_probs[n], Yte, thr)
    row.update(thr_val=thr, dice_tuned=round(mt["dice"], 4), iou_tuned=round(mt["iou"], 4))
    # (A) test-time augmentation
    _, ma = evaluate(predict(trained[n], dl_te, tta=True), Yte)
    row.update(dice_tta=round(ma["dice"], 4), iou_tta=round(ma["iou"], 4))
    # (C) cross-dataset generalisation
    if dl_cvc is not None:
        _, mc = evaluate(predict(trained[n], dl_cvc), Ycvc)
        row.update(cvc_dice=round(mc["dice"], 4), cvc_iou=round(mc["iou"], 4))
    rows.append(row)
results = pd.DataFrame(rows).set_index("model")
print(results)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for n in MODELS:
    axs[0].plot(histories[n].epoch, histories[n].train_loss, label=n)
    axs[1].plot(histories[n].epoch, histories[n].val_dice, label=n)
axs[0].set(title="Training loss", xlabel="epoch"); axs[1].set(title="Validation Dice", xlabel="epoch")
axs[1].legend(); plt.tight_layout(); plt.savefig(f"{OUT}/figures/training_curves.png", dpi=130); plt.close()

# %% [markdown]
# ## 7. Accuracy vs inference speed (real-time requirement ≈ 25–30 FPS)

# %%
@torch.no_grad()
def fps(model, half, warmup=20, iters=200, batch=1):
    """Frames per second for one frame at a time. batch=4 = the 4 flipped copies used by TTA (still 1 frame)."""
    model.eval(); x = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    if SMOKE: warmup, iters = 2, 5
    for _ in range(warmup):
        with torch.autocast("cuda", enabled=half): model(x)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        with torch.autocast("cuda", enabled=half): model(x)
    if DEVICE == "cuda": torch.cuda.synchronize()
    return iters / (time.perf_counter() - t)

for n in MODELS:
    results.loc[n, "params_M"] = round(sum(p.numel() for p in trained[n].parameters()) / 1e6, 1)
    results.loc[n, "fps_fp32"] = round(fps(trained[n], False), 1)
    results.loc[n, "fps_fp16"] = round(fps(trained[n], DEVICE == "cuda"), 1)
    results.loc[n, "ms_per_frame_fp16"] = round(1000 / results.loc[n, "fps_fp16"], 2)
    results.loc[n, "fps_fp16_tta"] = round(fps(trained[n], DEVICE == "cuda", batch=len(FLIPS)), 1)
    results.loc[n, "train_min"] = round(train_minutes[n], 1)
results.to_csv(f"{OUT}/results.csv")
print(results)

plt.figure(figsize=(7, 5))
for n in MODELS:
    plt.scatter(results.loc[n, "fps_fp16"], results.loc[n, "dice"], s=40 + 4 * results.loc[n, "params_M"])
    plt.annotate(n, (results.loc[n, "fps_fp16"], results.loc[n, "dice"]), textcoords="offset points", xytext=(6, 6))
for n in MODELS:
    plt.scatter(results.loc[n, "fps_fp16_tta"], results.loc[n, "dice_tta"], marker="x", c="k", s=30)
    plt.plot([results.loc[n, "fps_fp16"], results.loc[n, "fps_fp16_tta"]],
             [results.loc[n, "dice"], results.loc[n, "dice_tta"]], c="gray", lw=0.6, ls=":")
plt.scatter([], [], marker="x", c="k", label="same model + TTA (4 flips)"); plt.legend(loc="lower right", fontsize=8)
plt.axvline(25, ls="--", c="gray"); plt.text(25.5, plt.ylim()[0] + 0.005, "25 FPS (real-time video)", color="gray")
plt.xlabel(f"FPS (batch 1, {IMG_SIZE}x{IMG_SIZE}, fp16, {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'})")
plt.ylabel("Test Dice"); plt.title("Accuracy vs speed (marker size ∝ parameters)")
plt.tight_layout(); plt.savefig(f"{OUT}/figures/accuracy_vs_speed.png", dpi=130); plt.close()

# %% [markdown]
# ## 8. Failure analysis: small and flat (low-contrast) polyps
# * **Size** = polyp area as a fraction of the image (from the original-resolution mask).
# * **Flat (real label)**: test images that appear in Kvasir-Sessile (196 sessile polyps ≤10 mm) are tagged sessile/flat.
# * **Flat proxy**: Kvasir-SEG itself has no "flat" label. Flat/sessile polyps barely stand out from the mucosa,
#   so we use *low colour contrast* between the polyp and a ring of surrounding tissue (ΔE in CIELAB) as a proxy.
#   This is an approximation and is reported as such.

# %%
def contrast(img, m):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    ring = cv2.dilate(m, np.ones((25, 25), np.uint8)) - m
    if m.sum() == 0 or ring.sum() == 0: return np.nan
    return float(np.linalg.norm(lab[m > 0].mean(0) - lab[ring > 0].mean(0)))

df = pd.DataFrame({"name": test_names, "area_frac": area_te, "contrast": [contrast(x, y) for x, y in zip(Xte, Yte)]})
for n in MODELS:
    df[f"dice_{n}"] = per_img[n]["dice"]; df[f"iou_{n}"] = per_img[n]["iou"]
df["size_bin"] = pd.cut(df.area_frac, [0, 0.05, 0.20, 1.0], labels=["small (<5%)", "medium (5-20%)", "large (>20%)"])
df["contrast_bin"] = pd.qcut(df.contrast, 3, labels=["low (flat proxy)", "mid", "high"]) if len(df) >= 6 else "n/a"
df["morphology"] = np.where(df["name"].isin(SESSILE), "sessile/flat (Kvasir-Sessile)", "other")
df.to_csv(f"{OUT}/per_image_test.csv", index=False)

def by_group(col):
    g = df.groupby(col, observed=False)
    t = pd.DataFrame({"n": g.size()})
    for n in MODELS:
        t[f"dice_{n}"] = g[f"dice_{n}"].mean().round(4)
        t[f"miss_rate_{n}"] = g[f"dice_{n}"].apply(lambda s: (s < 0.5).mean()).round(3)  # Dice<0.5 ≈ polyp missed
    return t
size_table, contrast_table, morph_table = by_group("size_bin"), by_group("contrast_bin"), by_group("morphology")
morph_table.to_csv(f"{OUT}/failure_by_morphology.csv"); print(morph_table)
size_table.to_csv(f"{OUT}/failure_by_size.csv"); contrast_table.to_csv(f"{OUT}/failure_by_contrast.csv")
print(size_table); print(contrast_table)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, t, title in [(axs[0], size_table, "Dice by polyp size"), (axs[1], contrast_table, "Dice by polyp contrast")]:
    w = 0.8 / len(MODELS); x = np.arange(len(t))
    for i, n in enumerate(MODELS):
        ax.bar(x + i * w, t[f"dice_{n}"], w, label=n)
    ax.set_xticks(x + w * (len(MODELS) - 1) / 2, [f"{k}\n(n={v})" for k, v in zip(t.index, t.n)])
    ax.set(title=title, ylabel="mean test Dice", ylim=(0, 1))
axs[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/figures/failure_by_size_contrast.png", dpi=130); plt.close()

plt.figure(figsize=(7, 5))
for n in MODELS: plt.scatter(df.area_frac * 100, df[f"dice_{n}"], s=12, alpha=0.6, label=n)
plt.xscale("log"); plt.xlabel("polyp area (% of image, log)"); plt.ylabel("Dice"); plt.legend(fontsize=8)
plt.title("Per-image Dice vs polyp size"); plt.tight_layout(); plt.savefig(f"{OUT}/figures/dice_vs_size.png", dpi=130); plt.close()

# %% [markdown]
# ## 9. Qualitative results

# %%
def overlay(img, m, color):
    o = img.copy(); o[m > 0] = (0.5 * o[m > 0] + 0.5 * np.array(color)).astype(np.uint8); return o

best_model = results.dice.idxmax()
worst = df.sort_values(f"dice_{best_model}").index[:8]
fig, axs = plt.subplots(len(worst), 3, figsize=(9, 3 * len(worst)))
axs = np.atleast_2d(axs)
for r, i in enumerate(worst):
    p = (test_probs[best_model][i, 0].numpy() > 0.5).astype(np.uint8)
    for c, (im, t) in enumerate([(Xte[i], f"{df["name"][i][:14]}  area {df.area_frac[i]*100:.1f}%"),
                                 (overlay(Xte[i], Yte[i], (0, 255, 0)), "ground truth"),
                                 (overlay(Xte[i], p, (255, 0, 0)), f"{best_model}  Dice {df[f'dice_{best_model}'][i]:.2f}")]):
        axs[r, c].imshow(im); axs[r, c].set_title(t, fontsize=8); axs[r, c].axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/figures/worst_cases_{best_model}.png", dpi=110); plt.close()

pick = df.sort_values("area_frac").index[np.linspace(0, len(df) - 1, min(5, len(df))).astype(int)]
fig, axs = plt.subplots(len(pick), 2 + len(MODELS), figsize=(2.6 * (2 + len(MODELS)), 2.6 * len(pick)))
axs = np.atleast_2d(axs)
for r, i in enumerate(pick):
    panels = [(Xte[i], "image"), (Yte[i], "ground truth")] + \
             [((test_probs[n][i, 0].numpy() > 0.5), f"{n}\nDice {df[f'dice_{n}'][i]:.2f}") for n in MODELS]
    for c, (im, t) in enumerate(panels):
        axs[r, c].imshow(im, cmap=None if im.ndim == 3 else "gray"); axs[r, c].set_title(t, fontsize=8); axs[r, c].axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/figures/model_comparison.png", dpi=110); plt.close()

# %% [markdown]
# ## 10. Attention-gate maps (Attention U-Net)
# Each gate outputs a weight in [0,1] per pixel that scales the encoder skip features.
# Bright = features passed to the decoder; dark = suppressed. Gate 1 is the coarsest level, gate 4 the finest.

# %%
att_model, att_maps = trained["attention_unet"].eval(), []
hooks = [g.psi.register_forward_hook(lambda m, i, o: att_maps.append(o.detach().float().cpu()))
         for g in att_model.gates]
idx = df.sort_values("area_frac").index[np.linspace(0, len(df) - 1, min(4, len(df))).astype(int)]
fig, axs = plt.subplots(len(idx), 6, figsize=(15, 2.7 * len(idx))); axs = np.atleast_2d(axs)
with torch.no_grad():
    for r, i in enumerate(idx):
        att_maps.clear()
        x = SegDS(Xte[i:i + 1], Yte[i:i + 1], False)[0][0][None].to(DEVICE)
        att_model(x)
        panels = [(Xte[i], f"image (area {df.area_frac[i]*100:.1f}%)"), (Yte[i], "ground truth")] + \
                 [(a[0, 0].numpy(), f"gate {k + 1} ({a.shape[-1]}px)") for k, a in enumerate(att_maps)]
        for c, (im, t) in enumerate(panels):
            axs[r, c].imshow(im, cmap=None if im.ndim == 3 else ("gray" if c == 1 else "jet"), vmin=None if c != 1 else 0)
            axs[r, c].set_title(t, fontsize=8); axs[r, c].axis("off")
for h in hooks: h.remove()
plt.tight_layout(); plt.savefig(f"{OUT}/figures/attention_maps.png", dpi=110); plt.close()

# %% [markdown]
# ## 11. Summary

# %%
summary = {"device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu", "img_size": IMG_SIZE,
           "split": {"train": len(train_names), "val": len(val_names), "test": len(test_names)},
           "results": results.reset_index().to_dict("records"),
           "failure_by_size": size_table.reset_index().astype(str).to_dict("records"),
           "failure_by_morphology": morph_table.reset_index().astype(str).to_dict("records"),
           "failure_by_contrast": contrast_table.reset_index().astype(str).to_dict("records"),
           "smoke": SMOKE}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
