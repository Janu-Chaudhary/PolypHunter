# 4. Results

<!-- Owner: Manoj (#6). Numbers come from results/summary/results_table.csv and confidence_intervals.csv,
     written by 04_analysis. `__` = waiting for the results of #4 (Trivedh). Bold the best value per column
     once all five rows are in. -->

## 4.1 Main results
Table 4.1 compares the five models on the same 120 test images. All numbers are means over images: each
image is scored on its own and then averaged, so a small polyp counts as much as a large one (Section 3.1).
"Dice @0.5" uses the default probability threshold of 0.5. "Dice tuned" uses the threshold that was best on
the **validation** set (Section 3.5). "Dice TTA" averages the predictions of four flipped views of each image
(test-time augmentation). The speed columns are explained in Section 5.

**Table 4.1.** Test results on Kvasir-SEG (n = 120). Best value per column in bold.

| Model | Params (M) | Dice @0.5 | IoU @0.5 | Precision | Recall | Miss rate | Threshold | Dice tuned | Dice TTA | FPS (fp16) |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net (scratch) | 31.0 | 0.851 | 0.774 | 0.867 | 0.897 | 4.2% | 0.50 | 0.851 | 0.863 | 45.7 |
| U-Net + ResNet-34 | __ | __ | __ | __ | __ | __ | __ | __ | __ | __ |
| U-Net + MobileNetV2 | __ | __ | __ | __ | __ | __ | __ | __ | __ | __ |
| Attention U-Net | 31.4 | 0.854 | 0.775 | 0.868 | 0.895 | 5.0% | 0.55 | 0.854 | 0.857 | 40.7 |
| SegFormer (MiT-B2) | 24.7 | 0.917 | 0.865 | 0.911 | 0.943 | 2.5% | 0.60 | 0.918 | 0.926 | 43.3 |

"Miss rate" is the share of test images with Dice below 0.5, where the polyp was essentially not found.

What the table shows:
- **Baseline.** The U-Net trained from scratch already reaches a Dice of 0.851 and an IoU of 0.774. Its recall
  (0.897) is higher than its precision (0.867): it tends to over-segment a little, marking some healthy tissue
  around the polyp, rather than missing parts of the polyp.
- **Pretrained encoder.** __ (ResNet-34 vs scratch: +__ Dice. Does ImageNet pretraining help with only 800
  training images, and how many epochs did it need compared with 60 for the baseline?)
- **Attention gates.** The Attention U-Net differs from the baseline only by its four attention gates, and it
  scores almost the same: Dice 0.854 against 0.851, IoU 0.775 against 0.774. It misses 6 polyps instead of 5,
  and the gates (0.4 M extra parameters, plus their extra computation at every skip connection) make it
  slightly slower (40.7 against 45.7 frames per second).
- **Transformer encoder.** SegFormer is the most accurate model: Dice 0.917 and IoU 0.865, 0.066 Dice above the
  baseline, with 6 M fewer parameters. Both its precision (0.911) and its recall (0.943) are higher, so it
  outlines polyps more tightly *and* finds more of each polyp. It misses only 3 of the 120 polyps.
- **Threshold tuning.** The best validation thresholds were 0.50, 0.55 and 0.60, and tuning changed test Dice by
  at most 0.001 (Section 3.5). The models are well calibrated around 0.5, so tuning is not a useful source of gain.
- **Test-time augmentation** raised the baseline's Dice by 0.012 (0.851 → 0.863) and SegFormer's by 0.009
  (0.917 → 0.926), but the Attention U-Net's by only 0.003 (0.854 → 0.857). It costs four forward passes per
  frame; Section 5 discusses whether that is worth it.

## 4.2 How certain are these numbers?
120 test images are a sample: on another 120 polyps every mean would come out a little different. We estimate
how much with the **bootstrap** (Efron and Tibshirani, 1993): we draw 120 images from the test set *with
replacement*, recompute the mean Dice, repeat this 2000 times and report the middle 95% of these means as the
**95% confidence interval** (CI).

To decide whether a model really beats the baseline, we bootstrap the **per-image difference** in Dice between
the two models. Because both models are scored on the same images, an unusually easy or hard image affects both
equally, so this paired comparison is much more sensitive than comparing two separate intervals. A gain counts
as significant when its 95% interval does not contain zero (two-sided p < 0.05).

**Table 4.2.** Mean test Dice with 95% bootstrap CI, and the gain over the U-Net baseline (paired bootstrap).

| Model | Dice [95% CI] | Gain over baseline [95% CI] | p |
|---|---|---|---|
| U-Net (scratch) | 0.851 [0.816, 0.883] | (baseline) | |
| U-Net + ResNet-34 | __ | __ | __ |
| U-Net + MobileNetV2 | __ | __ | __ |
| Attention U-Net | 0.854 [0.822, 0.882] | +0.003 [−0.011, +0.018] | 0.73 |
| SegFormer (MiT-B2) | 0.917 [0.893, 0.937] | +0.066 [+0.042, +0.093] | < 0.001 |

The baseline's interval is about ±0.034 wide, so with 120 test images a single mean Dice is only known to within
about three hundredths. Even so, the paired comparison separates the models clearly:
- **SegFormer's gain is significant.** Its gain interval lies entirely above zero; even its lower end (+0.042) is
  a clear improvement. In none of the 2000 resamples was SegFormer worse than the baseline.
- **The attention gates' gain is not.** The interval for the Attention U-Net's gain, −0.011 to +0.018, contains
  zero (p = 0.73): on this test set, the gates make no measurable difference.
- __ (ResNet-34 and MobileNetV2.)

These intervals cover the choice of test images only, not the randomness of training (Section 8.2).

## 4.3 Training curves
Figure 4.1 (`figures/training_curves.png`) shows the training loss and the validation Dice of every model per
epoch. The two models trained from scratch learn slowly: the baseline reached its best validation Dice (0.855)
at epoch 54 of 60, the Attention U-Net (0.859) at epoch 47, and both needed about 39 epochs to come within 2% of
their best. SegFormer, whose encoder was pretrained on ImageNet, was within 2% of its best after only 8 epochs and
peaked at epoch 23 of 40 (0.913). It trained in 12 minutes, against 46 for the baseline and 54 for the Attention
U-Net. No model overfitted: the validation Dice at the last epoch was within 0.005 of each model's best.
__ (ResNet-34 and MobileNetV2: do they also converge as fast as SegFormer?)

## 4.4 Qualitative comparison
Figure 4.2 (`figures/model_comparison.png`) shows five test polyps, from the smallest to the largest, with the
ground truth and the mask predicted by each model. __ (Describe what is visible: e.g. which models miss the
smallest polyp, which ones produce cleaner borders on large polyps, any false positives on glare or folds.)

## 4.5 Summary
SegFormer is the most accurate model by a clear, statistically significant margin (Dice 0.917, +0.066 over the
baseline), while attention gates add nothing measurable to a U-Net trained from scratch. All three models run
above 25 frames per second. __ (Where ResNet-34 and MobileNetV2 fit: is any CNN close to SegFormer, and which
model gives the best accuracy for its speed?) Section 6 shows where the remaining errors are.
