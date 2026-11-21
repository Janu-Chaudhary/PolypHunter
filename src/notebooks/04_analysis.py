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
MISS_DICE = 0.5                                      # an image with Dice below this counts as a missed polyp
N_BOOT = 2000                                        # bootstrap resamples; 10000 moves the CI ends by only ~0.002
BASELINE = "unet_scratch"                            # every other model is compared against this one
SMALL, LARGE = 0.05, 0.20                            # size bins as polyp share of the image: small < 5%, large > 20%

# Readable names for figures and tables, in the same order as ORDER.
LABELS = {"unet_scratch": "U-Net (scratch)", "unet_resnet34": "U-Net + ResNet-34",
          "unet_mobilenetv2": "U-Net + MobileNetV2", "attention_unet": "Attention U-Net",
          "segformer_mitb2": "SegFormer (MiT-B2)"}
# One fixed colour per model, the same in every figure, so a reader can follow a model across the report.
# The five hues were checked for colour-blind separation; three of them are light, so every figure
# also names the models in a legend or a direct label instead of relying on colour alone.
COLORS = {"unet_scratch": "#2a78d6", "unet_resnet34": "#eb6834", "unet_mobilenetv2": "#1baf7a",
          "attention_unet": "#eda100", "segformer_mitb2": "#e87ba4"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e4e1"   # text, secondary text and reference lines, grid
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.facecolor": "white"})

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

def save(fig, name, dpi=150):
    """Save a figure into FIGURES as name and close it, so figures do not pile up in memory."""
    fig.savefig(os.path.join(FIGURES, name), dpi=dpi, bbox_inches="tight"); plt.close(fig)

# %% [markdown]
# ## 1. Main results table
# Dice and IoU at the default 0.5 threshold, after validation-tuned thresholding and with TTA; the miss rate;
# speed on the GPU at batch 1; and Dice on CVC-ClinicDB (another hospital, no retraining).

# %%
COLS = ["dice", "iou", "precision", "recall", "thr_val", "dice_tuned", "dice_tta", "cvc_dice", "cvc_iou",
        "params_M", "fps_fp32", "fps_fp16", "fps_fp16_tta", "train_min"]
table = pd.DataFrame([{c: MET[m].get(c, np.nan) for c in COLS} for m in MODELS], index=MODELS).round(4)
# miss rate over all 120 test images: the share where the polyp was essentially not found
table.insert(4, "miss_rate", [(DF[f"dice_{m}"] < MISS_DICE).mean().round(4) for m in MODELS])
table["ms_per_frame_fp16"] = (1000 / table["fps_fp16"]).round(2)
table["realtime_fp16"] = table["fps_fp16"] >= REALTIME_FPS
table.insert(0, "model", [LABELS[m] for m in MODELS])
table.to_csv(os.path.join(SUMMARY, "results_table.csv"))
open(os.path.join(SUMMARY, "results_table.txt"), "w").write(table.to_string())   # readable copy for the report
print(table.to_string())

# %% [markdown]
# ## 1b. 95% confidence intervals (bootstrap over test images)
# With only 120 test images, a mean Dice could be a little higher or lower on another 120 polyps. The bootstrap
# estimates how much: draw 120 images *with replacement* from the test set, recompute the mean, repeat 2000
# times, and keep the middle 95% of those means as the interval.
#
# To compare a model with the baseline we resample the **per-image difference** (model Dice - baseline Dice on
# the same image). This is a paired comparison: both models are scored on the same polyps, so an easy or hard
# image affects both equally and only the real difference between the models is left.
# `p_value` (two-sided) is twice the share of resamples on the less common side of zero. Below 0.05 (the same as
# the gain's interval not containing 0) means the difference is unlikely to come from which images happened to
# land in the test set.
#
# These intervals cover the choice of test images only. They do not cover training randomness (one seed per model).

# %%
def bootstrap_ci(values, idx):
    """Mean of values and its 95% bootstrap interval.

    values: one number per test image (N,); idx: (N_BOOT, N) resample indices, each row one resample.
    Returns (mean, lower 2.5% percentile, upper 97.5% percentile, the N_BOOT resampled means)."""
    boot = values[idx].mean(axis=1)                  # one mean per resample
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return values.mean(), lo, hi, boot

# Each row of IDX is one resample of the test set. The same rows are used for every model, so the
# comparisons between models are paired image by image.
IDX = np.random.default_rng(42).integers(0, len(DF), size=(N_BOOT, len(DF)))
base = BASELINE if BASELINE in MODELS else MODELS[0]
ci_rows = []
for m in MODELS:
    d, lo, hi, _ = bootstrap_ci(DF[f"dice_{m}"].to_numpy(), IDX)
    i, ilo, ihi, _ = bootstrap_ci(DF[f"iou_{m}"].to_numpy(), IDX)
    row = {"model": LABELS[m], "dice": d, "dice_lo": lo, "dice_hi": hi, "iou": i, "iou_lo": ilo, "iou_hi": ihi}
    if m != base:
        diff = DF[f"dice_{m}"].to_numpy() - DF[f"dice_{base}"].to_numpy()     # per-image gain over the baseline
        g, glo, ghi, gboot = bootstrap_ci(diff, IDX)
        # two-sided: the model could also be worse than the baseline; capped at 1 when the gain is ~0
        p = min(1.0, 2 * min((gboot <= 0).mean(), (gboot >= 0).mean()))
        row.update(gain_vs_baseline=g, gain_lo=glo, gain_hi=ghi, p_value=p)
    ci_rows.append(row)
CI_TABLE = pd.DataFrame(ci_rows, index=MODELS).round(4)
CI_TABLE.to_csv(os.path.join(SUMMARY, "confidence_intervals.csv"))
print(CI_TABLE.to_string())

# %% [markdown]
# ## 2. Training curves
# Left: training loss per epoch. Right: validation Dice per epoch, the number used to keep the best epoch.

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for m in MODELS:
    axs[0].plot(HIST[m].epoch, HIST[m].train_loss, lw=2, color=COLORS[m], label=LABELS[m])
    axs[1].plot(HIST[m].epoch, HIST[m].val_dice, lw=2, color=COLORS[m], label=LABELS[m])
axs[0].set(title="Training loss (BCE + Dice)", xlabel="epoch", ylabel="loss")
axs[1].set(title="Validation Dice", xlabel="epoch", ylabel="Dice")
axs[1].legend(loc="lower right")
fig.tight_layout(); save(fig, "training_curves.png")

# %% [markdown]
# ## 3. Accuracy vs speed
# One point per model, named next to it; marker area grows with the number of parameters.
# The vertical bar is the 95% confidence interval of the Dice (section 1b): models whose bars overlap a lot
# may not really differ in accuracy. Points right of the dashed line run in real time.

# %%
def label_side(m):
    """'left' if another model sits close to the right of model m (its label would run into it), else 'right'.

    A neighbour is in the way when it is within 10% of the x range to the right and either its point is within
    8% of the y range or its error bar spans the label's height."""
    xs = {k: MET[k]["fps_fp16"] for k in MODELS}; ys = {k: MET[k]["dice"] for k in MODELS}
    xr = max(xs.values()) * 1.3                          # same as the x limit set below
    yr = max(CI_TABLE.dice_hi) - min(CI_TABLE.dice_lo)   # the error bars set the y range
    def in_way(k):
        """True if model k's point or error bar would sit under model m's label."""
        close_x = 0 <= xs[k] - xs[m] < 0.10 * xr
        bar_covers = CI_TABLE.dice_lo[k] <= ys[m] <= CI_TABLE.dice_hi[k]
        return close_x and (abs(ys[k] - ys[m]) < 0.08 * yr or bar_covers)
    return "left" if any(in_way(k) for k in MODELS if k != m) else "right"

fig, ax = plt.subplots(figsize=(7.5, 5))
for m in MODELS:
    x, y = MET[m]["fps_fp16"], MET[m]["dice"]
    ax.errorbar(x, y, yerr=[[y - CI_TABLE.dice_lo[m]], [CI_TABLE.dice_hi[m] - y]], color=COLORS[m],
                lw=1.5, capsize=4, zorder=2)
    ax.scatter(x, y, s=40 + 4 * MET[m]["params_M"], color=COLORS[m], edgecolor="white", linewidth=1.5,
               label=f"{LABELS[m]} ({MET[m]['params_M']:.1f} M params)", zorder=3)
    left = label_side(m) == "left"
    ax.annotate(LABELS[m], (x, y), textcoords="offset points", xytext=(-9 if left else 9, 6),
                ha="right" if left else "left", fontsize=8, color=INK)
ax.axvline(REALTIME_FPS, ls="--", lw=1, c=MUTED)
ax.text(REALTIME_FPS, 1.01, f" {REALTIME_FPS} FPS: real-time video", transform=ax.get_xaxis_transform(),
        color=MUTED, fontsize=8)
ax.set_xlim(0, 1.3 * max(MET[m]["fps_fp16"] for m in MODELS))   # 30% headroom so the rightmost label fits
ax.set(xlabel=f"frames per second (batch 1, fp16, {MET[MODELS[0]].get('gpu', 'GPU')})", ylabel="test Dice")
ax.set_title("Accuracy vs inference speed (bars: 95% CI of Dice; marker size ~ parameters)", pad=16)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, markerscale=0.6)   # below the plot, never on a point
fig.tight_layout(); save(fig, "accuracy_vs_speed.png")

# %% [markdown]
# ## 4. Failure analysis: small, flat and low-contrast polyps
# * **Size** = polyp area / image area. Bins: small < 5%, medium 5-20%, large > 20%.
# * **Morphology**: test images listed in Kvasir-Sessile (expert-labelled sessile polyps <= 10 mm) are "sessile/flat".
# * **Contrast** (stand-in for flatness): colour distance between polyp and surrounding tissue; lowest third = "low".
# * **Miss rate** = share of images with Dice < 0.5, i.e. the polyp was essentially not found.
# * **Size x morphology**: Kvasir-Sessile only holds polyps under 10 mm, so "small" and "flat" overlap; this table
#   separates the two effects.

# %%
DF["size_bin"] = pd.cut(DF.area_frac, [0, SMALL, LARGE, 1.0],
                        labels=["small (<5%)", "medium (5-20%)", "large (>20%)"])
DF["morphology"] = np.where(DF.sessile == 1, "sessile/flat", "other")
DF["contrast_bin"] = pd.qcut(DF.contrast, 3, labels=["low", "mid", "high"])     # thirds of the test set
DF["small"] = np.where(DF.area_frac < SMALL, "small (<5%)", "not small")

def by_group(cols):
    """Mean Dice and miss rate (Dice < MISS_DICE) per model for each value of the column(s) cols, with group size n."""
    g = DF.groupby(cols, observed=False)
    t = pd.DataFrame({"n": g.size()})
    for m in MODELS:
        t[f"dice_{m}"] = g[f"dice_{m}"].mean().round(4)
        t[f"miss_{m}"] = g[f"dice_{m}"].apply(lambda s: (s < MISS_DICE).mean()).round(3)
    return t

FAIL = {"size": by_group("size_bin"), "morphology": by_group("morphology"), "contrast": by_group("contrast_bin")}
for k, t in FAIL.items():
    t.to_csv(os.path.join(SUMMARY, f"failure_by_{k}.csv"))
    print(f"\n--- by {k} ---\n{t.to_string()}")
CROSS = by_group(["morphology", "small"])            # 4 cells: other/sessile x not small/small
CROSS.to_csv(os.path.join(SUMMARY, "failure_by_size_morphology.csv"))
print(f"\n--- by morphology and size ---\n{CROSS.to_string()}")

fig, axs = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
w = 0.8 / len(MODELS)                                   # bar width so all models fit in one group
for ax, (k, t) in zip(axs, FAIL.items()):
    x = np.arange(len(t))
    for i, m in enumerate(MODELS):
        # white edge = a thin gap between neighbouring bars, so similar colours stay apart
        ax.bar(x + i * w, t[f"dice_{m}"], w, color=COLORS[m], edgecolor="white", linewidth=1.5, label=LABELS[m])
    ax.set_xticks(x + w * (len(MODELS) - 1) / 2, [f"{a}\n(n={n})" for a, n in zip(t.index, t.n)])
    ax.set(title=f"Dice by polyp {k}", ylim=(0, 1))
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("mean test Dice")
fig.tight_layout(rect=(0, 0, 1, 0.92))                # leave the top 8% free for the shared legend
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=len(MODELS), fontsize=9)
save(fig, "failure_analysis.png")

# Small multiples, one panel per model: 5 x 120 points in one panel would hide each other.
flat = DF.sessile == 1                                  # True for the Kvasir-Sessile test images
fig, axs = plt.subplots(1, len(MODELS), figsize=(3.4 * len(MODELS), 3.6), sharex=True, sharey=True, squeeze=False)
for ax, m in zip(axs[0], MODELS):
    ax.scatter(DF.area_frac[~flat] * 100, DF[f"dice_{m}"][~flat], s=16, color=COLORS[m], alpha=0.75,
               edgecolor="none", label="other")
    ax.scatter(DF.area_frac[flat] * 100, DF[f"dice_{m}"][flat], s=22, marker="^", facecolor="none",
               edgecolor=COLORS[m], linewidth=1.2, label="sessile/flat")
    ax.axvline(SMALL * 100, ls="--", lw=1, c=MUTED)     # left of this line = small polyps
    ax.axhline(MISS_DICE, ls=":", lw=1, c=MUTED)        # below this line = missed polyp
    ax.set_xscale("log"); ax.set(title=LABELS[m], xlabel="polyp area (% of image)", ylim=(-0.03, 1.03))
    # log scale spreads out the many small polyps; plain-number ticks instead of 10^x keep it readable
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
axs[0, 0].set_ylabel("Dice per test image")
axs[0, 0].legend(loc="lower right")
fig.suptitle("Per-image Dice vs polyp size (dashed: 5% size cut-off; dotted: miss threshold)", y=1.02)
fig.tight_layout(); save(fig, "dice_vs_size.png")
DF.to_csv(os.path.join(SUMMARY, "per_image_all_models.csv"), index=False)

# %% [markdown]
# ## 5. Generalisation to CVC-ClinicDB
# The drop from Kvasir test Dice to CVC Dice shows how much each model depends on one hospital's images.

# %%
if all("cvc_dice" in MET[m] for m in MODELS):
    gen = pd.DataFrame({"model": [LABELS[m] for m in MODELS],
                        "kvasir_dice": [MET[m]["dice"] for m in MODELS], "cvc_dice": [MET[m]["cvc_dice"] for m in MODELS],
                        "kvasir_iou": [MET[m]["iou"] for m in MODELS], "cvc_iou": [MET[m]["cvc_iou"] for m in MODELS]},
                       index=MODELS)
    gen["drop_dice"] = gen.kvasir_dice - gen.cvc_dice
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
elif not glob.glob(os.path.join(found[0], DF["name"].iloc[0] + ".*")):
    # fake results use invented names (fake_000, ...), which have no matching image in the real dataset
    print("test image names not in the dataset (fake results?): galleries skipped")
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
        for c, (im, t) in enumerate([(img, f"{row['name']}\narea {row.area_frac * 100:.1f}%, {row.morphology}"),
                                     (overlay(img, gt, (0, 255, 0)), "ground truth"),
                                     (overlay(img, pred, (255, 0, 0)), f"{LABELS[best]}, Dice {row[f'dice_{best}']:.2f}")]):
            axs[r, c].imshow(im); axs[r, c].set_title(t, fontsize=8); axs[r, c].axis("off")
    fig.tight_layout(); save(fig, "worst_cases.png", dpi=110)

    pick = DF.sort_values("area_frac").iloc[np.linspace(0, len(DF) - 1, 5).astype(int)]   # smallest .. largest
    fig, axs = plt.subplots(len(pick), 2 + len(MODELS), figsize=(2.6 * (2 + len(MODELS)), 2.6 * len(pick)))
    for r, (_, row) in enumerate(pick.iterrows()):
        img, gt = load_test_image(*found, row["name"], size)
        panels = [(img, f"image, area {row.area_frac * 100:.1f}%", None), (gt, "ground truth", "gray")]
        panels += [(probs[m][row["name"]] > 127, f"{LABELS[m]}\nDice {row[f'dice_{m}']:.2f}", "gray") for m in MODELS]
        for c, (im, t, cmap) in enumerate(panels):
            axs[r, c].imshow(im, cmap=cmap); axs[r, c].set_title(t, fontsize=8); axs[r, c].axis("off")
    fig.tight_layout(); save(fig, "model_comparison.png", dpi=110)

print("tables in", SUMMARY, "| figures in", FIGURES)
