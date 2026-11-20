# %% [markdown]
# ## Shared code (src/polyphunter/common.py — do not edit inside the notebook)

# %%
"""
PolypHunter shared code: everything the three training notebooks have in common.

It finds the datasets, makes the fixed train/val/test split, loads and augments images, defines the
models, trains one model, evaluates it, measures its speed and writes its results in the team's
output format (results/<model>/). Notebooks 01-03 only choose which models to train and add their
own figures. This file is inlined at the top of every built notebook by tools/make_notebook.py,
so Kaggle needs no extra files.

The file is frozen: change it only through a PR announced to the whole team.
"""
import os, glob, json, time, random
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp

# ---------------- configuration ----------------
# SMOKE is set by the notebook's first cell before this code runs. True = tiny 1-epoch run that only
# checks the pipeline; its outputs go to results_smoke/ so they can never be committed by mistake.
SMOKE = globals().get("SMOKE", False)
IMG_SIZE = 352                      # every image is resized to 352x352; divisible by 16 for 4 poolings
BATCH_SIZE = 8                      # fits a 16 GB T4 for all five models with mixed precision
SEED = 42                           # one seed for split, weights and augmentation so runs are comparable
N_TEST, N_VAL = 120, 80             # 1000 images -> 800 train / 80 val / 120 test (test size as in Jha et al. 2020)
EPOCHS = {"unet_scratch": 60, "attention_unet": 60,          # from scratch: needs more epochs
          "unet_resnet34": 40, "unet_mobilenetv2": 40, "segformer_mitb2": 40}   # pretrained: converges faster
LR = {"unet_scratch": 3e-4, "attention_unet": 3e-4, "unet_resnet34": 2e-4,
      "unet_mobilenetv2": 3e-4, "segformer_mitb2": 1e-4}    # smaller LR keeps pretrained weights from being destroyed
THRESHOLDS = np.round(np.arange(0.20, 0.81, 0.05), 2)       # probability cut-offs searched on the validation set
FLIPS = [[], [3], [2], [2, 3]]      # TTA views: identity, horizontal flip (dim 3), vertical flip (dim 2), both
MEAN = np.array([0.485, 0.456, 0.406], np.float32)          # ImageNet statistics, used for every model so
STD = np.array([0.229, 0.224, 0.225], np.float32)           # pretrained encoders see the input they expect
if SMOKE:
    EPOCHS = {k: 1 for k in EPOCHS}

ON_KAGGLE = os.path.isdir("/kaggle/working")
OUT = "/kaggle/working" if ON_KAGGLE else os.path.abspath(".")
RESULTS = os.path.join(OUT, "results_smoke" if SMOKE else "results")   # results/<model>/ lives here
FIGURES = os.path.join(OUT, "figures")
WEIGHTS = os.path.join(OUT, "weights")                                  # never committed (see .gitignore)
for d in (RESULTS, FIGURES, WEIGHTS):
    os.makedirs(d, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
print("torch", torch.__version__, "| smp", smp.__version__, "| device", GPU_NAME, "| SMOKE", SMOKE)


def seed_all(s=SEED):
    """Seed Python, NumPy and PyTorch so a rerun gives the same split, initial weights and augmentations."""
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------- data ----------------
def find_kvasir(root):
    """Return (images_dir, masks_dir) of Kvasir-SEG under root, or None.

    Kvasir-SEG keeps images/ and masks/ side by side. Requiring at least 900 images skips the
    196-image Kvasir-Sessile folder, which has the same layout."""
    for img_dir in glob.glob(f"{root}/**/images", recursive=True):
        mask_dir = os.path.join(os.path.dirname(img_dir), "masks")
        if os.path.isdir(mask_dir) and len(os.listdir(img_dir)) >= 900:
            return img_dir, mask_dir
    return None


def find_sessile_ids():
    """Return the image ids in Kvasir-Sessile: 196 Kvasir-SEG polyps that experts labelled sessile (flat) and <=10 mm.

    These ids give the failure analysis a real "flat polyp" label instead of a guess."""
    files = glob.glob("/kaggle/input/**/sessile-main-Kvasir-SEG/images/*", recursive=True)
    return {os.path.splitext(os.path.basename(f))[0] for f in files}


def read_pair(img_dir, mask_dir, name):
    """Read one RGB image and its binary mask (1 = polyp pixel) at original resolution."""
    ip = glob.glob(os.path.join(img_dir, name + ".*"))[0]
    mp = glob.glob(os.path.join(mask_dir, name + ".*"))[0]
    img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    return img, (m > 127).astype(np.uint8)          # masks are JPEGs, so threshold away compression noise


def resize_pair(img, m):
    """Resize an image (area interpolation, keeps detail) and its mask (nearest, keeps it binary) to IMG_SIZE."""
    return (cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA),
            cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST))


def load_set(img_dir, mask_dir, names):
    """Load and resize a list of images into RAM.

    Returns X (N,H,W,3 uint8), Y (N,H,W uint8) and area (N,): the polyp's share of the image measured
    on the original-resolution mask, used later for the small-polyp analysis."""
    X, Y, area = [], [], []
    for n in names:
        img, m = read_pair(img_dir, mask_dir, n)
        area.append(m.mean())
        x, y = resize_pair(img, m)
        X.append(x); Y.append(y)
    return np.stack(X), np.stack(Y), np.array(area)


def load_cvc():
    """Load CVC-ClinicDB (612 frames from another hospital) for the cross-dataset test, or (None, None)."""
    for orig in glob.glob("/kaggle/input/**/Original", recursive=True):
        gt = os.path.join(os.path.dirname(orig), "Ground Truth")
        if not os.path.isdir(gt):
            continue
        X, Y = [], []
        for f in sorted(os.listdir(orig)):
            g = glob.glob(os.path.join(gt, os.path.splitext(f)[0] + ".*"))
            img = cv2.imread(os.path.join(orig, f))
            if not g or img is None:
                continue
            m = (cv2.imread(g[0], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            x, y = resize_pair(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), m)
            X.append(x); Y.append(y)
        if len(X) > 100:
            print("CVC-ClinicDB:", len(X), "images from", orig)
            return np.stack(X), np.stack(Y)
    print("CVC-ClinicDB not found: cross-dataset test skipped")
    return None, None


def augment(img, m):
    """Random training augmentation applied identically to image and mask.

    Flips and 90-degree rotations are free (colonoscopy has no 'up'); a small rotate/scale/shift and a
    brightness/contrast change imitate camera angle and lighting. This matters because 800 images is small."""
    if random.random() < 0.5: img, m = img[:, ::-1], m[:, ::-1]
    if random.random() < 0.5: img, m = img[::-1], m[::-1]
    k = random.randint(0, 3); img, m = np.rot90(img, k), np.rot90(m, k)
    img, m = np.ascontiguousarray(img), np.ascontiguousarray(m)
    if random.random() < 0.7:
        ang, sc = random.uniform(-25, 25), random.uniform(0.85, 1.15)
        M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), ang, sc)
        M[:, 2] += np.random.uniform(-0.06, 0.06, 2) * IMG_SIZE         # shift by up to 6% of the image
        img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = cv2.warpAffine(m, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    if random.random() < 0.6:
        a, b = random.uniform(0.8, 1.2), random.uniform(-25, 25)        # contrast gain a, brightness offset b
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return img, m


class SegDS(torch.utils.data.Dataset):
    """PyTorch dataset over preloaded arrays: returns a normalised (3,H,W) image and a (1,H,W) float mask."""

    def __init__(self, X, Y, train):
        """X, Y: uint8 arrays from load_set; train=True turns augmentation on."""
        self.X, self.Y, self.train = X, Y, train

    def __len__(self):
        """Number of images."""
        return len(self.X)

    def __getitem__(self, i):
        """Return image i (augmented if training), normalised with ImageNet mean/std, and its mask."""
        img, m = self.X[i], self.Y[i]
        if self.train:
            img, m = augment(img, m)
        img = (img.astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(img.transpose(2, 0, 1)), torch.from_numpy(m[None].astype(np.float32))


def loader(X, Y, train):
    """DataLoader for one split. Training shuffles and drops the last partial batch (BatchNorm needs >1 sample)."""
    return torch.utils.data.DataLoader(SegDS(X, Y, train), batch_size=BATCH_SIZE, shuffle=train,
                                       num_workers=2, drop_last=train, pin_memory=True)


def prepare_data():
    """Find the datasets, make the fixed split, load everything and build the loaders.

    Returns a dict D with: names per split, X/Y arrays per split, area_te (test polyp areas),
    sessile (set of flat-polyp ids), Xcvc/Ycvc, and loaders dl_tr, dl_va, dl_te, dl_cvc."""
    seed_all()
    found = find_kvasir("/kaggle/input") if os.path.isdir("/kaggle/input") else None
    if found is None:
        found = find_kvasir("data")
    assert found, "Kvasir-SEG not found: add the dataset debeshjha1/kvasirseg to the notebook"
    img_dir, mask_dir = found
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png")))
    # Fixed split: shuffle the sorted names with SEED, then cut. Every member gets the same 120 test images.
    perm = np.random.RandomState(SEED).permutation(names).tolist()
    split = {"test": perm[:N_TEST], "val": perm[N_TEST:N_TEST + N_VAL], "train": perm[N_TEST + N_VAL:]}
    if SMOKE:
        split = {"train": split["train"][:32], "val": split["val"][:16], "test": split["test"][:16]}
    print(f"{len(names)} images | train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])}")
    D = {"names": split, "img_dir": img_dir, "mask_dir": mask_dir, "sessile": find_sessile_ids()}
    print(len(D["sessile"]), "sessile (flat) polyp ids found")
    D["Xtr"], D["Ytr"], _ = load_set(img_dir, mask_dir, split["train"])
    D["Xva"], D["Yva"], _ = load_set(img_dir, mask_dir, split["val"])
    D["Xte"], D["Yte"], D["area_te"] = load_set(img_dir, mask_dir, split["test"])
    D["Xcvc"], D["Ycvc"] = load_cvc()
    if SMOKE and D["Xcvc"] is not None:
        D["Xcvc"], D["Ycvc"] = D["Xcvc"][:16], D["Ycvc"][:16]
    D["dl_tr"] = loader(D["Xtr"], D["Ytr"], True)
    D["dl_va"] = loader(D["Xva"], D["Yva"], False)
    D["dl_te"] = loader(D["Xte"], D["Yte"], False)
    D["dl_cvc"] = loader(D["Xcvc"], D["Ycvc"], False) if D["Xcvc"] is not None else None
    return D


# ---------------- models ----------------
class DoubleConv(nn.Sequential):
    """U-Net building block: two 3x3 conv + BatchNorm + ReLU. padding=1 keeps height and width unchanged."""

    def __init__(self, cin, cout):
        """cin: input channels, cout: output channels."""
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class AttentionGate(nn.Module):
    """Attention gate (Oktay et al. 2018): alpha = sigmoid(psi(relu(Wg*g + Wx*x))), output = x * alpha.

    g is the decoder signal (coarse, knows *what* a polyp looks like), x is the encoder skip feature
    (fine, knows *where* edges are). alpha in [0,1] per pixel suppresses background before the skip is used."""

    def __init__(self, cg, cx, cint):
        """cg, cx: channels of g and x; cint: size of the shared intermediate space (half of cx here)."""
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(cg, cint, 1, bias=False), nn.BatchNorm2d(cint))
        self.wx = nn.Sequential(nn.Conv2d(cx, cint, 1, bias=False), nn.BatchNorm2d(cint))
        self.psi = nn.Sequential(nn.Conv2d(cint, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())   # psi -> one alpha per pixel

    def forward(self, g, x):
        """Return the gated skip features x * alpha (g and x have the same height and width here)."""
        a = self.psi(F.relu(self.wg(g) + self.wx(x)))
        return x * a


class UNet(nn.Module):
    """U-Net (Ronneberger et al. 2015) with optional attention gates on the skip connections.

    Encoder: 5 DoubleConv levels with 64->1024 channels; 4 max-poolings shrink 352 -> 22 px.
    Decoder: 4 transposed convs double the size back to 352; each concatenates the matching encoder
    feature (the skip connection) so fine detail lost by pooling returns. A 1x1 conv gives 1 logit per pixel."""

    def __init__(self, ch=(64, 128, 256, 512, 1024), attention=False):
        """ch: channels per level; attention=True builds the Attention U-Net."""
        super().__init__()
        self.attention = attention
        self.enc = nn.ModuleList([DoubleConv(3, ch[0])] + [DoubleConv(ch[i], ch[i + 1]) for i in range(4)])
        # decoder goes from the deepest level (index 3) back to the first (index 0)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(ch[i] * 2, ch[i]) for i in reversed(range(4))])   # *2: skip + upsampled
        if attention:
            self.gates = nn.ModuleList([AttentionGate(ch[i], ch[i], ch[i] // 2) for i in reversed(range(4))])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        """x: (B,3,H,W) normalised images -> (B,1,H,W) logits (apply sigmoid for probabilities)."""
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))    # no pooling before the first level
            skips.append(x)
        x = skips.pop()                                    # deepest feature (22x22, 1024 ch) starts the decoder
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x); s = skips.pop()                     # s: encoder feature at the same resolution
            if self.attention:
                s = self.gates[j](x, s)                    # gate the skip with the decoder signal
            x = dec(torch.cat([s, x], 1))
        return self.head(x)


class Resize(nn.Module):
    """Wrapper that upsamples logits to the input size; SegFormer's head predicts at 1/4 resolution."""

    def __init__(self, m):
        """m: the wrapped model."""
        super().__init__()
        self.m = m

    def forward(self, x):
        """Run the model and bilinearly resize its logits to x's height and width if they differ."""
        y = self.m(x)
        return y if y.shape[-2:] == x.shape[-2:] else F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build(name):
    """Create a fresh model by its key. Pretrained encoders download ImageNet weights (needs Internet On)."""
    if name == "unet_scratch": return UNet()
    if name == "attention_unet": return UNet(attention=True)
    if name == "unet_resnet34": return smp.Unet("resnet34", encoder_weights="imagenet", classes=1)
    if name == "unet_mobilenetv2": return smp.Unet("mobilenet_v2", encoder_weights="imagenet", classes=1)
    if name == "segformer_mitb2":
        if hasattr(smp, "Segformer"):                      # smp >= 0.4 has the SegFormer decoder
            return Resize(smp.Segformer("mit_b2", encoder_weights="imagenet", classes=1))
        return Resize(smp.FPN("mit_b2", encoder_weights="imagenet", classes=1))
    raise ValueError(name)


def n_params(model):
    """Number of parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


# ---------------- loss and metrics ----------------
def dice_loss(logits, y, eps=1.0):
    """Soft Dice loss, 1 - Dice on probabilities. Counters class imbalance: a polyp is often <20% of the pixels."""
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (den + eps)).mean()


def loss_fn(logits, y):
    """Training loss = BCE (per-pixel accuracy) + Dice loss (overlap of the whole polyp)."""
    return F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)


def per_image_metrics(pred, y, eps=1e-7):
    """Dice, IoU, precision and recall for each image.

    pred, y: binary tensors (B,1,H,W). tp/fp/fn are counted per image, so every image weighs the same
    in the mean regardless of polyp size. Dice = 2TP/(2TP+FP+FN), IoU = TP/(TP+FP+FN)."""
    pred, y = pred.float(), y.float()
    tp = (pred * y).sum((1, 2, 3)); fp = (pred * (1 - y)).sum((1, 2, 3)); fn = ((1 - pred) * y).sum((1, 2, 3))
    return {"dice": ((2 * tp + eps) / (2 * tp + fp + fn + eps)).cpu().numpy(),
            "iou": ((tp + eps) / (tp + fp + fn + eps)).cpu().numpy(),
            "precision": ((tp + eps) / (tp + fp + eps)).cpu().numpy(),
            "recall": ((tp + eps) / (tp + fn + eps)).cpu().numpy()}


@torch.no_grad()
def predict(model, dl, tta=False):
    """Return polyp probabilities (N,1,H,W) on the CPU for every image in dl.

    tta=True averages the predictions of the 4 flipped views (each flipped back first): test-time
    augmentation, usually a little more accurate but 4x the compute."""
    model.eval(); probs = []
    for x, _ in dl:
        x = x.to(DEVICE)
        with torch.autocast("cuda", enabled=DEVICE == "cuda"):
            p = 0
            for f in (FLIPS if tta else [[]]):
                pf = torch.sigmoid(model(x.flip(f) if f else x).float())
                p = p + (pf.flip(f) if f else pf)
            p = p / (len(FLIPS) if tta else 1)
        probs.append(p.cpu())
    return torch.cat(probs)


def evaluate(probs, Y, thr=0.5):
    """Binarise probabilities at thr and score them against masks Y (N,H,W). Returns (per-image dict, means dict)."""
    m = per_image_metrics(probs > thr, torch.from_numpy(Y[:, None]))
    return m, {k: float(v.mean()) for k, v in m.items()}


# ---------------- training ----------------
def train(name, D):
    """Train model `name` on D's training set and keep the epoch with the best validation Dice.

    AdamW + OneCycle learning-rate schedule (warm-up for 10% of steps, then cosine decay), mixed precision.
    Model selection uses validation Dice only; the test set is never looked at during training.
    Returns (best model, history DataFrame, training minutes)."""
    seed_all()
    model = build(name).to(DEVICE)
    ep = EPOCHS[name]
    opt = torch.optim.AdamW(model.parameters(), lr=LR[name], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR[name], total_steps=ep * len(D["dl_tr"]), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    hist, best, t0 = [], -1, time.time()
    wpath = os.path.join(WEIGHTS, f"{name}.pt")
    for e in range(ep):
        model.train(); tl = 0
        for x, y in D["dl_tr"]:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                loss = loss_fn(model(x).float(), y)       # loss in fp32 for numerical stability
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(x)
        _, vm = evaluate(predict(model, D["dl_va"]), D["Yva"])
        hist.append({"epoch": e + 1, "train_loss": tl / (len(D["dl_tr"]) * BATCH_SIZE),
                     "val_dice": vm["dice"], "val_iou": vm["iou"]})
        if vm["dice"] > best:
            best = vm["dice"]; torch.save(model.state_dict(), wpath)
        print(f"[{name}] ep {e + 1:3d}/{ep} loss {hist[-1]['train_loss']:.4f} "
              f"val dice {vm['dice']:.4f} iou {vm['iou']:.4f} (best {best:.4f})")
    model.load_state_dict(torch.load(wpath, map_location=DEVICE))
    minutes = (time.time() - t0) / 60
    print(f"[{name}] done in {minutes:.1f} min")
    return model, pd.DataFrame(hist), minutes


@torch.no_grad()
def fps(model, half, batch=1, warmup=20, iters=200):
    """Frames per second for one frame at a time, the way live colonoscopy video arrives.

    Warm-up runs first (cuDNN picks its algorithms, memory is allocated), and torch.cuda.synchronize()
    brackets the timed loop because GPU calls return before the work is done. batch=4 times the four
    TTA views of one frame, so the result is still frames per second."""
    model.eval(); x = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    if SMOKE:
        warmup, iters = 2, 5
    for _ in range(warmup):
        with torch.autocast("cuda", enabled=half):
            model(x)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        with torch.autocast("cuda", enabled=half):
            model(x)
    if DEVICE == "cuda": torch.cuda.synchronize()
    return iters / (time.perf_counter() - t)


# ---------------- results in the team's output format ----------------
def contrast(img, m):
    """Colour contrast of a polyp: distance (Delta E) in CIELAB between the polyp's mean colour and a 25-px ring
    of tissue around it. Low values = polyp blends into the wall, a stand-in for flat lesions."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    ring = cv2.dilate(m, np.ones((25, 25), np.uint8)) - m          # 12 px on each side of the border
    if m.sum() == 0 or ring.sum() == 0:
        return float("nan")
    return float(np.linalg.norm(lab[m > 0].mean(0) - lab[ring > 0].mean(0)))


def write_shared(D):
    """Write results/split.json and results/test_attributes.csv (identical from every notebook).

    test_attributes.csv: name, area_frac (polyp share of the image), contrast, sessile (1 = in Kvasir-Sessile).
    The analysis notebook joins these with each model's per-image scores."""
    json.dump(D["names"], open(os.path.join(RESULTS, "split.json"), "w"), indent=1)
    pd.DataFrame({"name": D["names"]["test"], "area_frac": D["area_te"],
                  "contrast": [contrast(x, y) for x, y in zip(D["Xte"], D["Yte"])],
                  "sessile": [int(n in D["sessile"]) for n in D["names"]["test"]]}
                 ).to_csv(os.path.join(RESULTS, "test_attributes.csv"), index=False)


def run_model(name, D):
    """Train one model and write results/<name>/ in the team's output format. Returns (model, metrics dict).

    Files: metrics.json (scores, tuned threshold, TTA, CVC, speed, size), history.csv (per epoch),
    per_image_test.csv (scores per test image) and test_probs.npz (test probabilities as uint8 0-255)."""
    model, hist, minutes = train(name, D)
    out = os.path.join(RESULTS, name); os.makedirs(out, exist_ok=True)
    probs = predict(model, D["dl_te"])
    per_img, mean = evaluate(probs, D["Yte"])
    met = {"model": name, **mean}
    # Threshold tuning: choose the cut-off on validation only, then apply it once to the test set.
    vp = predict(model, D["dl_va"])
    val_curve = {str(t): evaluate(vp, D["Yva"], t)[1]["dice"] for t in THRESHOLDS}
    thr = float(max(val_curve, key=val_curve.get))
    _, mt = evaluate(probs, D["Yte"], thr)
    met.update(thr_val=thr, dice_tuned=mt["dice"], iou_tuned=mt["iou"], val_dice_by_threshold=val_curve)
    _, ma = evaluate(predict(model, D["dl_te"], tta=True), D["Yte"])
    met.update(dice_tta=ma["dice"], iou_tta=ma["iou"])
    if D["dl_cvc"] is not None:                          # cross-dataset test, no retraining
        _, mc = evaluate(predict(model, D["dl_cvc"]), D["Ycvc"])
        met.update(cvc_dice=mc["dice"], cvc_iou=mc["iou"])
    half = DEVICE == "cuda"
    met.update(params_M=n_params(model), fps_fp32=fps(model, False), fps_fp16=fps(model, half),
               fps_fp16_tta=fps(model, half, batch=len(FLIPS)), train_min=minutes, epochs=EPOCHS[name],
               lr=LR[name], gpu=GPU_NAME, smoke=SMOKE)
    json.dump(met, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    hist.to_csv(os.path.join(out, "history.csv"), index=False)
    pd.DataFrame({"name": D["names"]["test"], **per_img}).to_csv(os.path.join(out, "per_image_test.csv"), index=False)
    np.savez_compressed(os.path.join(out, "test_probs.npz"), names=np.array(D["names"]["test"]),
                        probs=(probs[:, 0].numpy() * 255).round().astype(np.uint8))   # uint8 keeps the file ~4x smaller
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()
                      if k != "val_dice_by_threshold"}, indent=1))
    torch.cuda.empty_cache()
    return model, met
