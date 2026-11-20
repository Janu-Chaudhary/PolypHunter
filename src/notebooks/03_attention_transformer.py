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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = prepare_data()
write_shared(D)
models, metrics = {}, {}
for name in ["attention_unet", "segformer_mitb2"]:
    models[name], metrics[name] = run_model(name, D)

# %% [markdown]
# ## Attention-gate maps (report section "Attention U-Net")
# Each gate outputs alpha in [0,1] per pixel, which scales the encoder skip features before the decoder uses them.
# Bright = passed on, dark = suppressed. Gate 1 works at 44x44 px (coarse), gate 4 at 352x352 px (fine).
# Rows go from the smallest to the largest polyp in the test set.

# %%
att_maps = []                                     # filled by the hooks: one alpha map per gate per forward pass
model = models["attention_unet"].eval()
hooks = [g.psi.register_forward_hook(lambda mod, inp, out: att_maps.append(out.detach().float().cpu()))
         for g in model.gates]
idx = np.argsort(D["area_te"])[np.linspace(0, len(D["area_te"]) - 1, min(4, len(D["area_te"]))).astype(int)]
fig, axs = plt.subplots(len(idx), 6, figsize=(15, 2.7 * len(idx)))
axs = np.atleast_2d(axs)
with torch.no_grad():
    for r, i in enumerate(idx):
        att_maps.clear()
        x = SegDS(D["Xte"][i:i + 1], D["Yte"][i:i + 1], False)[0][0][None].to(DEVICE)   # one normalised image
        model(x)
        panels = [(D["Xte"][i], f"image (polyp {D['area_te'][i] * 100:.1f}%)", None),
                  (D["Yte"][i], "ground truth", "gray")]
        panels += [(a[0, 0].numpy(), f"gate {k + 1} ({a.shape[-1]} px)", "jet") for k, a in enumerate(att_maps)]
        for c, (im, title, cmap) in enumerate(panels):
            axs[r, c].imshow(im, cmap=cmap); axs[r, c].set_title(title, fontsize=8); axs[r, c].axis("off")
for h in hooks:
    h.remove()
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "attention_maps.png"), dpi=110); plt.close()
print("saved", os.path.join(FIGURES, "attention_maps.png"))

# %% [markdown]
# ## Threshold-tuning curve (report section "Threshold tuning")
# Validation Dice for every cut-off between 0.20 and 0.80. The best one (marked) is the only one applied to
# the test set; choosing it on the test set would leak test information into the model.

# %%
plt.figure(figsize=(7, 4.5))
for name, m in metrics.items():
    t = [float(k) for k in m["val_dice_by_threshold"]]
    v = list(m["val_dice_by_threshold"].values())
    plt.plot(t, v, marker="o", ms=3, label=name)
    plt.scatter([m["thr_val"]], [m["val_dice_by_threshold"][str(m["thr_val"])]], s=80, facecolors="none", edgecolors="k")
plt.axvline(0.5, ls="--", c="gray")
plt.xlabel("probability threshold"); plt.ylabel("validation Dice"); plt.legend(fontsize=8)
plt.title("Threshold chosen on the validation set")
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "threshold_curve.png"), dpi=130); plt.close()
print("saved", os.path.join(FIGURES, "threshold_curve.png"))
