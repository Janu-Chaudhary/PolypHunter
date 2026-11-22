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
matplotlib.use("Agg")                 # draw figures to files only; Kaggle's batch run has no screen
import matplotlib.pyplot as plt

# D holds the fixed split, the images and masks of every split as arrays, and the four data loaders
# (see prepare_data in the shared code above).
D = prepare_data()
write_shared(D)                       # results/split.json and results/test_attributes.csv, shared by all notebooks
# train the from-scratch U-Net (60 epochs) and write results/unet_scratch/; metrics = the dict saved in metrics.json
model, metrics = run_model("unet_scratch", D)

# %% [markdown]
# ## Dataset sample figure (report section "Dataset")
# Four test images of increasing polyp size, each with its ground-truth mask, to show what the task looks like.

# %%
# argsort orders the 120 test images by polyp area; linspace picks 4 evenly spaced positions in that order
# (first, two in between, last), i.e. the smallest polyp, two medium ones and the largest
idx = np.argsort(D["area_te"])[np.linspace(0, len(D["area_te"]) - 1, 4).astype(int)]   # smallest .. largest polyp
fig, axs = plt.subplots(2, 4, figsize=(12, 6))      # row 0: images, row 1: masks; one column per chosen polyp
for c, i in enumerate(idx):
    axs[0, c].imshow(D["Xte"][i]); axs[0, c].set_title(f"polyp area {D['area_te'][i] * 100:.1f}%", fontsize=9)
    axs[1, c].imshow(D["Yte"][i], cmap="gray"); axs[1, c].set_title("ground-truth mask", fontsize=9)
for a in axs.flat:
    a.axis("off")                                   # images need no axis ticks
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "dataset_samples.png"), dpi=130); plt.close()
print("saved", os.path.join(FIGURES, "dataset_samples.png"))
