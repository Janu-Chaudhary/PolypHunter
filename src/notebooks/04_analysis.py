# %% [markdown]
# # PolypHunter 04: analysis of all models (owner: Manoj Kharkar, issue #6)
#
# Reads every `results/<model>/` folder written by notebooks 01-03 and builds the project's tables and
# figures: the main results table, training curves, accuracy vs speed, failure analysis on small /
# flat (sessile) / low-contrast polyps, cross-dataset generalisation (CVC-ClinicDB) and image galleries.
#
# Needs no GPU and no PyTorch. Develop it on fake results first (`python3 tools/make_fake_results.py`,
# then `RESULTS_DIR=results/_fake`), and rerun it on the real results once the training PRs are merged.
# The two image galleries also need the Kvasir-SEG images (attached on Kaggle, or unzipped under `data/`);
# without them they are skipped.

# %%
import os, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Where the per-model folders are. "results" = real results; "results/_fake" = fake ones for development.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
SUMMARY = os.path.join(RESULTS_DIR, "summary")       # tables written by this notebook
FIGURES = os.environ.get("FIGURES_DIR", "figures" if RESULTS_DIR == "results" else os.path.join(RESULTS_DIR, "figures"))
os.makedirs(SUMMARY, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
ORDER = ["unet_scratch", "unet_resnet34", "unet_mobilenetv2", "attention_unet", "segformer_mitb2"]   # brief's order
REALTIME_FPS = 25                                    # lower end of colonoscopy video frame rates

# %% [markdown]
# ## Load every model's results

# %%
def load_results(root):
    """Read results/<model>/ for every model found under root.

    Returns (models in ORDER, metrics dict per model, history DataFrame per model,
    per-image test DataFrame joined with the test attributes: area_frac, contrast, sessile)."""
    models = [m for m in ORDER if os.path.exists(os.path.join(root, m, "metrics.json"))]
    assert models, f"no results found in {root}: run tools/make_fake_results.py or merge the training PRs"
    met = {m: json.load(open(os.path.join(root, m, "metrics.json"))) for m in models}
    hist = {m: pd.read_csv(os.path.join(root, m, "history.csv")) for m in models}
    df = pd.read_csv(os.path.join(root, "test_attributes.csv"))
    for m in models:
        pi = pd.read_csv(os.path.join(root, m, "per_image_test.csv"))
        df = df.merge(pi[["name", "dice", "iou"]].rename(columns={"dice": f"dice_{m}", "iou": f"iou_{m}"}), on="name")
    return models, met, hist, df

MODELS, MET, HIST, DF = load_results(RESULTS_DIR)
print("models:", MODELS, "| test images:", len(DF))

# %% [markdown]
# ## 1. Main results table
# Dice and IoU at the default 0.5 threshold, after validation-tuned thresholding and with TTA; speed on the GPU
# at batch 1; and Dice on CVC-ClinicDB (another hospital, no retraining).

# %%
COLS = ["dice", "iou", "precision", "recall", "thr_val", "dice_tuned", "dice_tta", "cvc_dice", "cvc_iou",
        "params_M", "fps_fp32", "fps_fp16", "fps_fp16_tta", "train_min"]
table = pd.DataFrame([{c: MET[m].get(c, np.nan) for c in COLS} for m in MODELS], index=MODELS).round(4)
table["ms_per_frame_fp16"] = (1000 / table["fps_fp16"]).round(2)
table["realtime_fp16"] = table["fps_fp16"] >= REALTIME_FPS
table.to_csv(os.path.join(SUMMARY, "results_table.csv"))
open(os.path.join(SUMMARY, "results_table.txt"), "w").write(table.to_string())   # readable copy for the report
print(table.to_string())

# %% [markdown]
# ## 2. Training curves

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for m in MODELS:
    axs[0].plot(HIST[m].epoch, HIST[m].train_loss, label=m)
    axs[1].plot(HIST[m].epoch, HIST[m].val_dice, label=m)
axs[0].set(title="Training loss (BCE + Dice)", xlabel="epoch")
axs[1].set(title="Validation Dice", xlabel="epoch"); axs[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "training_curves.png"), dpi=130); plt.close()

# %% [markdown]
# ## 3. Accuracy vs speed
# One point per model; marker size grows with the number of parameters. Points right of the dashed line run in real time.

# %%
plt.figure(figsize=(7, 5))
for m in MODELS:
    x, y = MET[m]["fps_fp16"], MET[m]["dice"]
    plt.scatter(x, y, s=40 + 4 * MET[m]["params_M"])
    plt.annotate(m, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
plt.axvline(REALTIME_FPS, ls="--", c="gray")
plt.text(REALTIME_FPS * 1.02, plt.ylim()[0] + 0.005, f"{REALTIME_FPS} FPS (real-time video)", color="gray", fontsize=8)
plt.xlabel(f"frames per second (batch 1, fp16, {MET[MODELS[0]].get('gpu', 'GPU')})"); plt.ylabel("test Dice")
plt.title("Accuracy vs inference speed (marker size ~ parameters)")
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "accuracy_vs_speed.png"), dpi=130); plt.close()

# %% [markdown]
# ## 4. Failure analysis: small, flat and low-contrast polyps
# * **Size** = polyp area / image area. Bins: small < 5%, medium 5-20%, large > 20%.
# * **Morphology**: test images listed in Kvasir-Sessile (expert-labelled sessile polyps <= 10 mm) are "sessile/flat".
# * **Contrast** (stand-in for flatness): colour distance between polyp and surrounding tissue; lowest third = "low".
# * **Miss rate** = share of images with Dice < 0.5, i.e. the polyp was essentially not found.

# %%
DF["size_bin"] = pd.cut(DF.area_frac, [0, 0.05, 0.20, 1.0], labels=["small (<5%)", "medium (5-20%)", "large (>20%)"])
DF["morphology"] = np.where(DF.sessile == 1, "sessile/flat", "other")
DF["contrast_bin"] = pd.qcut(DF.contrast, 3, labels=["low", "mid", "high"])     # thirds of the test set

def by_group(col):
    """Mean Dice and miss rate (Dice < 0.5) per model for each value of column col, with the group size n."""
    g = DF.groupby(col, observed=False)
    t = pd.DataFrame({"n": g.size()})
    for m in MODELS:
        t[f"dice_{m}"] = g[f"dice_{m}"].mean().round(4)
        t[f"miss_{m}"] = g[f"dice_{m}"].apply(lambda s: (s < 0.5).mean()).round(3)
    return t

FAIL = {"size": by_group("size_bin"), "morphology": by_group("morphology"), "contrast": by_group("contrast_bin")}
for k, t in FAIL.items():
    t.to_csv(os.path.join(SUMMARY, f"failure_by_{k}.csv"))
    print(f"\n--- by {k} ---\n{t.to_string()}")

fig, axs = plt.subplots(1, 3, figsize=(18, 4.5))
w = 0.8 / len(MODELS)                                   # bar width so all models fit in one group
for ax, (k, t) in zip(axs, FAIL.items()):
    x = np.arange(len(t))
    for i, m in enumerate(MODELS):
        ax.bar(x + i * w, t[f"dice_{m}"], w, label=m)
    ax.set_xticks(x + w * (len(MODELS) - 1) / 2, [f"{a}\n(n={n})" for a, n in zip(t.index, t.n)])
    ax.set(title=f"Dice by polyp {k}", ylabel="mean test Dice", ylim=(0, 1))
axs[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "failure_analysis.png"), dpi=130); plt.close()

plt.figure(figsize=(7, 5))
for m in MODELS:
    plt.scatter(DF.area_frac * 100, DF[f"dice_{m}"], s=12, alpha=0.6, label=m)
plt.xscale("log"); plt.xlabel("polyp area (% of image, log scale)"); plt.ylabel("Dice")
plt.title("Per-image Dice vs polyp size"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "dice_vs_size.png"), dpi=130); plt.close()
DF.to_csv(os.path.join(SUMMARY, "per_image_all_models.csv"), index=False)

# %% [markdown]
# ## 5. Generalisation to CVC-ClinicDB
# The drop from Kvasir test Dice to CVC Dice shows how much each model depends on one hospital's images.

# %%
if all("cvc_dice" in MET[m] for m in MODELS):
    gen = pd.DataFrame({"kvasir_dice": [MET[m]["dice"] for m in MODELS], "cvc_dice": [MET[m]["cvc_dice"] for m in MODELS]},
                       index=MODELS)
    gen["drop"] = gen.kvasir_dice - gen.cvc_dice
    gen.round(4).to_csv(os.path.join(SUMMARY, "generalisation_cvc.csv"))
    print(gen.round(4).to_string())
else:
    print("CVC results missing for some models: generalisation table skipped")

# %% [markdown]
# ## 6. Image galleries (need the Kvasir-SEG images)
# Worst eight test images of the best model, and a side-by-side comparison of all models on five polyps from
# small to large. Red = prediction, green = ground truth.

# %%
def find_kvasir_images():
    """Return (images_dir, masks_dir) of Kvasir-SEG on Kaggle or under data/, or None (galleries are skipped)."""
    for root in ("/kaggle/input", "data"):
        for img_dir in glob.glob(f"{root}/**/images", recursive=True):
            mask_dir = os.path.join(os.path.dirname(img_dir), "masks")
            if os.path.isdir(mask_dir) and len(os.listdir(img_dir)) >= 900:
                return img_dir, mask_dir
    return None

def load_test_image(img_dir, mask_dir, name, size):
    """Read one test image and its mask resized to the prediction size (size x size)."""
    import cv2                                           # only needed for the galleries
    img = cv2.cvtColor(cv2.imread(glob.glob(os.path.join(img_dir, name + ".*"))[0]), cv2.COLOR_BGR2RGB)
    m = cv2.imread(glob.glob(os.path.join(mask_dir, name + ".*"))[0], cv2.IMREAD_GRAYSCALE) > 127
    return (cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA),
            cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST))

def overlay(img, m, color):
    """Blend a colour into the pixels where mask m is 1 (50% image, 50% colour)."""
    o = img.copy(); o[m > 0] = (0.5 * o[m > 0] + 0.5 * np.array(color)).astype(np.uint8)
    return o

found = find_kvasir_images()
if found is None:
    print("Kvasir-SEG images not found: galleries skipped (attach the dataset on Kaggle or unzip it under data/)")
else:
    probs = {}
    for m in MODELS:
        z = np.load(os.path.join(RESULTS_DIR, m, "test_probs.npz"))
        probs[m] = dict(zip(z["names"], z["probs"]))     # name -> uint8 probability map (0-255)
    size = next(iter(probs[MODELS[0]].values())).shape[0]
    best = max(MODELS, key=lambda m: MET[m]["dice"])
    worst = DF.sort_values(f"dice_{best}").head(8)
    fig, axs = plt.subplots(len(worst), 3, figsize=(9, 3 * len(worst)))
    for r, (_, row) in enumerate(worst.iterrows()):
        img, gt = load_test_image(*found, row["name"], size)
        pred = probs[best][row["name"]] > 127            # 127 of 255 = probability 0.5
        for c, (im, t) in enumerate([(img, f"area {row.area_frac * 100:.1f}%  {row.morphology}"),
                                     (overlay(img, gt, (0, 255, 0)), "ground truth"),
                                     (overlay(img, pred, (255, 0, 0)), f"{best}  Dice {row[f'dice_{best}']:.2f}")]):
            axs[r, c].imshow(im); axs[r, c].set_title(t, fontsize=8); axs[r, c].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "worst_cases.png"), dpi=110); plt.close()

    pick = DF.sort_values("area_frac").iloc[np.linspace(0, len(DF) - 1, 5).astype(int)]
    fig, axs = plt.subplots(len(pick), 2 + len(MODELS), figsize=(2.6 * (2 + len(MODELS)), 2.6 * len(pick)))
    for r, (_, row) in enumerate(pick.iterrows()):
        img, gt = load_test_image(*found, row["name"], size)
        panels = [(img, "image", None), (gt, "ground truth", "gray")]
        panels += [(probs[m][row["name"]] > 127, f"{m}\nDice {row[f'dice_{m}']:.2f}", "gray") for m in MODELS]
        for c, (im, t, cmap) in enumerate(panels):
            axs[r, c].imshow(im, cmap=cmap); axs[r, c].set_title(t, fontsize=8); axs[r, c].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "model_comparison.png"), dpi=110); plt.close()

print("tables in", SUMMARY, "| figures in", FIGURES)
