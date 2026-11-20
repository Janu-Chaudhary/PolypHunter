# %% [markdown]
# # PolypHunter 01: U-Net baseline (owner: Janu Chaudhary, issue #3)
#
# Trains the U-Net of Ronneberger et al. (2015) from scratch on Kvasir-SEG. It is the reference point
# that every other model is compared against. Also writes the shared files `results/split.json` and
# `results/test_attributes.csv`, plus a dataset sample figure for the report.
#
# **Kaggle settings:** GPU T4, Internet On, inputs `debeshjha1/kvasirseg`, `debeshjha1/kvasirsessile`,
# `balraj98/cvcclinicdb`. About 30 minutes. Download `results/` and `figures/` from the Output tab when done.

# %%
# True = 1-epoch pipeline check on a few images (outputs go to results_smoke/). Set False for the real run.
SMOKE = False

# %%
# segmentation-models-pytorch provides the pretrained encoders; Kaggle does not ship it.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=False)

# %% COMMON

# %% [markdown]
# ## Load data and train

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = prepare_data()
write_shared(D)
model, metrics = run_model("unet_scratch", D)

# %% [markdown]
# ## Dataset sample figure (report section "Dataset")
# Four test images of increasing polyp size, each with its ground-truth mask, to show what the task looks like.

# %%
idx = np.argsort(D["area_te"])[np.linspace(0, len(D["area_te"]) - 1, 4).astype(int)]   # smallest .. largest polyp
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
for c, i in enumerate(idx):
    axs[0, c].imshow(D["Xte"][i]); axs[0, c].set_title(f"polyp area {D['area_te'][i] * 100:.1f}%", fontsize=9)
    axs[1, c].imshow(D["Yte"][i], cmap="gray"); axs[1, c].set_title("ground-truth mask", fontsize=9)
for a in axs.flat:
    a.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "dataset_samples.png"), dpi=130); plt.close()
print("saved", os.path.join(FIGURES, "dataset_samples.png"))
