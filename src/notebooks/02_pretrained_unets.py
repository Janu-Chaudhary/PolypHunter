# %% [markdown]
# # PolypHunter 02: pretrained U-Nets and the speed/TTA study (owner: Challa Trivedh Kumar, issue #4)
#
# Trains two U-Nets whose encoders were pretrained on ImageNet (transfer learning):
# * `unet_resnet34`: ResNet-34 encoder, the "U-Net with pretrained ResNet encoder" step of the brief.
# * `unet_mobilenetv2`: MobileNetV2 encoder (depthwise-separable convolutions), the lightweight real-time point.
#
# Speed is measured by `fps()` in the shared code (batch 1, 352x352, T4, fp32/fp16, with and without TTA).
# The extra figure shows what test-time augmentation (TTA) buys in Dice and costs in speed.
#
# **Kaggle settings:** GPU T4, Internet On, inputs `debeshjha1/kvasirseg`, `debeshjha1/kvasirsessile`,
# `balraj98/cvcclinicdb`. About 20 minutes. Download `results/` and `figures/` from the Output tab when done.

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
metrics = {}
for name in ["unet_resnet34", "unet_mobilenetv2"]:
    _, metrics[name] = run_model(name, D)

# %% [markdown]
# ## TTA trade-off figure (report section "Accuracy vs speed")
# Each arrow goes from a model without TTA to the same model with TTA: right-to-left = slower, upward = more accurate.
# The dashed line is 25 FPS, the lower end of colonoscopy video frame rates.

# %%
plt.figure(figsize=(7, 5))
for name, m in metrics.items():
    plt.annotate("", xy=(m["fps_fp16_tta"], m["dice_tta"]), xytext=(m["fps_fp16"], m["dice"]),
                 arrowprops=dict(arrowstyle="->", color="gray"))
    plt.scatter(m["fps_fp16"], m["dice"], s=50, label=f"{name}")
    plt.scatter(m["fps_fp16_tta"], m["dice_tta"], s=50, marker="x", c="k")
    plt.annotate(name, (m["fps_fp16"], m["dice"]), textcoords="offset points", xytext=(6, 6), fontsize=8)
plt.scatter([], [], marker="x", c="k", label="same model + TTA (4 flips)")
plt.axvline(25, ls="--", c="gray")
plt.xlabel(f"frames per second (batch 1, {IMG_SIZE}x{IMG_SIZE}, fp16, {GPU_NAME})"); plt.ylabel("test Dice")
plt.title("Test-time augmentation: accuracy gained vs speed lost"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "tta_tradeoff.png"), dpi=130); plt.close()
print("saved", os.path.join(FIGURES, "tta_tradeoff.png"))
