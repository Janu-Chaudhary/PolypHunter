# 5. Accuracy vs inference speed

A segmentation model is only useful at the bedside if it can keep up with live colonoscopy video, which
arrives at **25-30 frames per second**. This section measures whether our five models can, and what it
costs to make them more accurate.

### Measurement protocol (`fps()` in `common.py`)
- **Batch size 1.** Live video arrives one frame at a time, not as a batch, so batching would overstate
  real-world throughput.
- **Input 352×352**, the same resolution every model trains and evaluates at, run on a Kaggle **T4** GPU.
- **20 warm-up calls, not timed.** The first calls to a GPU model are slow: CUDA allocates memory and cuDNN
  benchmarks several convolution algorithms before picking the fastest one. Timing these would measure
  one-off setup cost, not steady-state speed.
- **200 timed calls**, with `torch.cuda.synchronize()` immediately before and after the timed block. GPU
  calls are asynchronous — the Python call returns as soon as the work is *queued*, not when it finishes —
  so without synchronising, the measured time would be wrong (usually far too short).
- Reported as **`fps_fp32`**, **`fps_fp16`** (mixed precision, half the memory and roughly 2-4× faster on a
  T4's tensor cores) and **`fps_fp16_tta`** (fp16 with test-time augmentation, see below).
- **ms per frame = 1000 / FPS** — the number that actually determines whether a frame is processed before
  the next one arrives.

### Test-time augmentation (TTA)
For each frame we predict on the original image and on its horizontal flip, vertical flip, and both flips
combined, flip each prediction back to the original orientation, and average the four. Averaging smooths
out orientation-dependent errors, so it usually improves Dice a little (`dice_tta`) — but it means **four
forward passes instead of one**, so throughput drops by roughly, though not exactly, a factor of four (some
of the per-call overhead is shared). `figures/tta_tradeoff.png` plots this trade-off for our two models: an
arrow from the no-TTA point to the TTA point of the same model, right-to-left meaning slower and upward
meaning more accurate.

### Results across all five models

| Model | Params (M) | FPS fp32 | FPS fp16 | FPS fp16+TTA | ms/frame (fp16) | Dice | Dice+TTA |
|---|---|---|---|---|---|---|---|
| U-Net (scratch) | 31.0 | 11.6 | 45.7 | 14.7 | 21.9 | 0.851 | 0.863 |
| U-Net + ResNet-34 | 24.4 | 44.9 | 109.5 | 48.7 | 9.1 | 0.903 | 0.912 |
| U-Net + MobileNetV2 | 6.6 | 86.5 | 91.7 | 63.1 | 10.9 | 0.894 | 0.900 |
| Attention U-Net | __ | __ | __ | __ | __ | __ | __ |
| SegFormer (MiT-B2) | __ | __ | __ | __ | __ | __ | __ |

(Attention U-Net and SegFormer rows come from `results/summary/results_table.csv` once Debangsu's results
are merged; see Section 4 for the consolidated table with every model.)

### Which models reach 25 FPS
Without TTA, all three models measured here clear 25 FPS comfortably, including the from-scratch baseline
(45.7 FPS) — the bottleneck for real-time use is **not** raw model speed at this resolution, it is whether
we can afford TTA's 4× cost. With TTA, the picture changes: the baseline drops to 14.7 FPS, **below** the
real-time threshold, while both pretrained models stay above it — ResNet-34 at 48.7 FPS and MobileNetV2 at
63.1 FPS. MobileNetV2 keeps the largest margin above 25 FPS in every configuration, which matters in
practice because a deployed system also spends time on video decoding, pre/post-processing and display, not
just the model's forward pass — the raw FPS number is a ceiling, not what the clinician actually gets.

### Is TTA worth it?
For both pretrained models, TTA buys roughly **0.006-0.009 Dice** (ResNet-34: 0.903 → 0.912; MobileNetV2:
0.894 → 0.900) at the cost of 2.2-2.3× fewer frames per second. Given that both models still clear 25 FPS
with TTA enabled, and the real-time budget has margin to spare (especially for MobileNetV2), this is a
trade worth taking here — unlike for the from-scratch baseline, where TTA pushes it below the real-time
threshold entirely.

### Recommendation
For deployment in a real-time colonoscopy system, we recommend **U-Net + MobileNetV2, with TTA enabled**:
it is the lightest model (6.6 M parameters, roughly a quarter of ResNet-34's and a fifth of the baseline's),
keeps the largest speed margin above the 25-30 FPS requirement even with TTA's four forward passes, and its
Dice (0.900 with TTA) trails ResNet-34 by only about one point while leaving far more headroom for the
non-model costs of a real system. ResNet-34 remains the better choice where maximum accuracy matters more
than speed margin, for example in offline or second-read review rather than live guidance.
