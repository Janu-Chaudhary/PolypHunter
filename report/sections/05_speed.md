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
- Reported as **`fps_fp32`**, **`fps_fp16`** (mixed precision) and **`fps_fp16_tta`** (fp16 with test-time
  augmentation, see below). Mixed precision is not a uniform win: see "Does fp16 always help?" below — the
  gain depends on the architecture, and for one model (SegFormer) fp16 is actually slower.
- **ms per frame = 1000 / FPS** — the number that actually determines whether a frame is processed before
  the next one arrives.

### Test-time augmentation (TTA)
For each frame we predict on the original image and on its horizontal flip, vertical flip, and both flips
combined, flip each prediction back to the original orientation, and average the four. Averaging smooths
out orientation-dependent errors, so it usually improves Dice a little (`dice_tta`). `fps()` times this as
**one batch of 4** (the four flipped views go through the model together), not as four separate forward
passes, so the throughput cost is not a fixed ~4× — it depends on how well each architecture's layers batch,
and ranges from 1.3× to 3.2× across our five models (see below). `figures/tta_tradeoff.png` plots this
trade-off for our two pretrained models: an arrow from the no-TTA point to the TTA point of the same model,
right-to-left meaning slower and upward meaning more accurate. (The figure only covers the two models
trained in this notebook; extending it to all five is an analysis-stage follow-up, not part of this section.)

### Results across all five models

| Model | Params (M) | FPS fp32 | FPS fp16 | FPS fp16+TTA | ms/frame (fp16) | Dice | Dice+TTA |
|---|---|---|---|---|---|---|---|
| U-Net (scratch) | 31.0 | 11.6 | 45.7 | 14.7 | 21.9 | 0.851 | 0.863 |
| U-Net + ResNet-34 | 24.4 | 44.9 | 109.5 | 48.7 | 9.1 | 0.903 | 0.912 |
| U-Net + MobileNetV2 | 6.6 | 86.5 | 91.7 | 63.1 | 10.9 | 0.894 | 0.900 |
| Attention U-Net | 31.4 | 10.6 | 40.7 | 12.7 | 24.6 | 0.854 | 0.857 |
| SegFormer (MiT-B2) | 24.7 | 56.9 | 43.3 | 32.8 | 23.1 | 0.917 | 0.926 |

### Which models reach 25 FPS
**In fp16**, all five models clear 25 FPS without TTA, from the slowest (Attention U-Net, 40.7 FPS) to the
fastest (ResNet-34, 109.5 FPS) — at this resolution, fp16 throughput is not the bottleneck for any of them.
**In fp32**, the picture is different: the scratch U-Net (11.6 FPS) and Attention U-Net (10.6 FPS) fall well
short of 25 FPS, because their double-convolution blocks run at full precision get no help from tensor
cores; only ResNet-34 (44.9), MobileNetV2 (86.5) and SegFormer (56.9) clear it in fp32.

With TTA (always timed in fp16), the scratch baseline and Attention U-Net both drop below the real-time
threshold (14.7 and 12.7 FPS), while ResNet-34 (48.7), MobileNetV2 (63.1) and SegFormer (32.8) stay above
it. MobileNetV2 does **not** keep the largest margin in every configuration: in fp16 without TTA, ResNet-34
is faster (109.5 vs 91.7 FPS) — MobileNetV2 leads only in fp32 (86.5) and in fp16+TTA (63.1), where its
depthwise-separable blocks batch the four TTA views more cheaply. This matters in practice because a
deployed system also spends time on video decoding, pre/post-processing and display, not just the model's
forward pass — the raw FPS number is a ceiling, not what the clinician actually gets.

### Is TTA worth it?
TTA buys a small Dice gain for every model (+0.003 to +0.012) at a cost in frames per second that varies a
lot by architecture: 3.1× for the scratch baseline (45.7 → 14.7), 3.2× for Attention U-Net (40.7 → 12.7),
2.25× for ResNet-34 (109.5 → 48.7), 1.45× for MobileNetV2 (91.7 → 63.1) and 1.32× for SegFormer
(43.3 → 32.8). The two models whose fp16 speed already sits close to the 25 FPS floor (the scratch baseline
and Attention U-Net) are pushed below it by TTA; the other three stay above it, with MobileNetV2 keeping the
most headroom (63.1 FPS) and SegFormer the least (32.8 FPS, still about 31% above the floor).

### Does fp16 always help?
No — the gain is model-dependent. Going from fp32 to fp16 speeds up the standard-convolution U-Nets a lot
(scratch 3.9×, Attention U-Net 3.8×, ResNet-34 2.4×), but MobileNetV2 gains only 1.06× (86.5 → 91.7): its
depthwise convolutions make little use of tensor cores, and at this size per-layer overhead dominates.
SegFormer is **slower** in fp16 than in fp32 (56.9 → 43.3): mixed precision adds type casts, and its
attention and normalisation layers gain little from it. The practical conclusion is that each model should
be deployed in whichever precision is actually faster for it — fp16 for the four U-Net variants, but fp32
for SegFormer.

### Recommendation
Across all five models, two poles stand out. SegFormer (MiT-B2) is the most accurate model we trained —
Dice 0.917 [95% CI 0.893, 0.937], rising to 0.926 with TTA — and still clears the real-time floor with TTA
at 32.8 FPS (about 31% above 25 FPS), but it must be deployed in **fp32** (56.9 FPS without TTA) to reach
its best speed, not fp16. U-Net + MobileNetV2 trades some of that accuracy for raw speed headroom: Dice
0.894 [0.862, 0.921] (0.900 with TTA) but 63.1 FPS with TTA, at only 6.6 M parameters.

A paired bootstrap over the same 120 test images (2000 resamples, Section 4.2's method) shows SegFormer's
edge over MobileNetV2 is real (+0.023 Dice [0.008, 0.040], p = 0.005), and MobileNetV2 also has the highest
miss rate of the three fast models (5.0% vs SegFormer's 2.5%) and the largest accuracy drop of all five
models on the out-of-distribution CVC-ClinicDB set (0.141, more than the scratch baseline's 0.101) — a
generalisation concern beyond raw Dice.

U-Net + ResNet-34 sits between the two poles: Dice 0.903 [0.874, 0.928] (0.912 with TTA), not significantly
different from either SegFormer (+0.014 [-0.006, 0.036], p = 0.17) or MobileNetV2 (+0.009 [-0.009, 0.027],
p = 0.33), while reaching 48.7 FPS with TTA — 2.5× SegFormer's fp16 throughput (109.5 vs 43.3 FPS) — and
generalising to CVC-ClinicDB far better than MobileNetV2 (drop of 0.075 vs 0.141).

Our recommendation: **SegFormer, run in fp32, where accuracy is the priority**, and **U-Net + ResNet-34
where real-time speed matters most** — it is statistically indistinguishable from SegFormer on Dice while
being markedly faster and more robust on CVC-ClinicDB than MobileNetV2. We do not recommend MobileNetV2 for
deployment despite its FPS headroom, given its weaker generalisation and higher miss rate. (Section 8
carries the project-wide recommendation; it should match this comparison.)
