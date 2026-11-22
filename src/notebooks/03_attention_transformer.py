# %% [markdown]
# # PolypHunter 03: Attention U-Net and SegFormer (owner: Debangsu Misra, issue #5)
#
# The "attention U-Net / transformer encoder" step of the brief; we train both:
# * `attention_unet`: the same U-Net as the baseline plus attention gates on the four skip connections
#   (Oktay et al. 2018), trained from scratch, so any difference to the baseline comes from the gates.
# * `segformer_mitb2`: SegFormer with the MiT-B2 transformer encoder (Xie et al. 2021), ImageNet-pretrained.
#
# Extra figures: the attention-gate maps (where the model looks) and the threshold-tuning curve.
#
# **Kaggle settings:** GPU T4, Internet On, inputs `debeshjha1/kvasirseg`, `debeshjha1/kvasirsessile`,
# `balraj98/cvcclinicdb`. About 50 minutes. Download `results/` and `figures/` from the Output tab when done.

# %%
# True = 1-epoch pipeline check on a few images (outputs go to results_smoke/). Set False for the real run.
SMOKE = False

# %%
# segmentation-models-pytorch provides the pretrained encoders; Kaggle does not ship it.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=False)

# %% COMMON

# %% [markdown]
# ## Load data and train both models

# %%
import matplotlib
matplotlib.use("Agg")                 # draw figures to files only; Kaggle's batch run has no screen
import matplotlib.pyplot as plt

# D holds the fixed split, the images and masks of every split as arrays, and the four data loaders
# (see prepare_data in the shared code above).
D = prepare_data()
write_shared(D)                       # results/split.json and results/test_attributes.csv, shared by all notebooks
models, metrics = {}, {}              # model key -> trained model / the dict saved in its metrics.json
for name in ["attention_unet", "segformer_mitb2"]:
    models[name], metrics[name] = run_model(name, D)   # the Attention U-Net is kept for the attention maps below

# %% [markdown]
# ## Attention-gate maps (report section "Attention U-Net")
# Each gate outputs alpha in [0,1] per pixel, which scales the encoder skip features before the decoder uses them.
# Bright = passed on, dark = suppressed. Gate 1 works at 44x44 px (coarse), gate 4 at 352x352 px (fine).
# Rows go from the smallest to the largest polyp in the test set.

# %%
att_maps = []                                     # filled by the hooks: one alpha map per gate per forward pass
model = models["attention_unet"].eval()
# A forward hook runs after a layer's forward pass and receives its output. Hooking each gate's psi block
# (1x1 conv + BatchNorm + sigmoid) captures alpha, shape (1, 1, h, w), without changing the model.
# The gates run deepest first, so att_maps fills in the order 44, 88, 176, 352 px.
hooks = [g.psi.register_forward_hook(lambda mod, inp, out: att_maps.append(out.detach().float().cpu()))
         for g in model.gates]
# 4 test images with evenly spaced polyp sizes, smallest to largest (min() guards the 16-image smoke run)
idx = np.argsort(D["area_te"])[np.linspace(0, len(D["area_te"]) - 1, min(4, len(D["area_te"]))).astype(int)]
fig, axs = plt.subplots(len(idx), 6, figsize=(15, 2.7 * len(idx)))   # columns: image, mask, gates 1-4
axs = np.atleast_2d(axs)                          # keep 2-D indexing axs[r, c] even with a single row
with torch.no_grad():
    for r, i in enumerate(idx):
        att_maps.clear()                          # keep only this image's four alpha maps
        # SegDS(...)[0] = (image, mask) of the one-image slice; [0] = image (3, 352, 352); [None] = batch axis
        x = SegDS(D["Xte"][i:i + 1], D["Yte"][i:i + 1], False)[0][0][None].to(DEVICE)   # one normalised image
        model(x)                                  # the hooks fill att_maps during this pass; the prediction is not needed
        panels = [(D["Xte"][i], f"image (polyp {D['area_te'][i] * 100:.1f}%)", None),
                  (D["Yte"][i], "ground truth", "gray")]
        # a[0, 0]: first image and only channel -> the (h, w) alpha map; "jet": blue = 0 (suppressed), red = 1
        panels += [(a[0, 0].numpy(), f"gate {k + 1} ({a.shape[-1]} px)", "jet") for k, a in enumerate(att_maps)]
        for c, (im, title, cmap) in enumerate(panels):
            axs[r, c].imshow(im, cmap=cmap); axs[r, c].set_title(title, fontsize=8); axs[r, c].axis("off")
for h in hooks:
    h.remove()                                    # detach the hooks so later forward passes are not recorded
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "attention_maps.png"), dpi=110); plt.close()
print("saved", os.path.join(FIGURES, "attention_maps.png"))

# %% [markdown]
# ## Threshold-tuning curve (report section "Threshold tuning")
# Validation Dice for every cut-off between 0.20 and 0.80. The best one (marked) is the only one applied to
# the test set; choosing it on the test set would leak test information into the model.

# %%
plt.figure(figsize=(7, 4.5))
for name, m in metrics.items():
    # val_dice_by_threshold maps each threshold (stored as text, e.g. "0.45") to its validation Dice
    t = [float(k) for k in m["val_dice_by_threshold"]]
    v = list(m["val_dice_by_threshold"].values())
    plt.plot(t, v, marker="o", ms=3, label=name)
    # an open black circle marks the chosen threshold (thr_val) on the curve
    plt.scatter([m["thr_val"]], [m["val_dice_by_threshold"][str(m["thr_val"])]], s=80, facecolors="none", edgecolors="k")
plt.axvline(0.5, ls="--", c="gray")               # the default threshold, for comparison
plt.xlabel("probability threshold"); plt.ylabel("validation Dice"); plt.legend(fontsize=8)
plt.title("Threshold chosen on the validation set")
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "threshold_curve.png"), dpi=130); plt.close()
print("saved", os.path.join(FIGURES, "threshold_curve.png"))
