# 4. Results

<!-- Owner: Manoj (#6). Numbers come from results/summary/results_table.csv, written by 04_analysis.
     `__` = waiting for the results of #4 (Trivedh) and #5 (Debangsu). -->

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
| Attention U-Net | __ | __ | __ | __ | __ | __ | __ | __ | __ | __ |
| SegFormer (MiT-B2) | __ | __ | __ | __ | __ | __ | __ | __ | __ | __ |

"Miss rate" is the share of test images with Dice below 0.5, where the polyp was essentially not found.

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
| Attention U-Net | __ | __ | __ |
| SegFormer (MiT-B2) | __ | __ | __ |

The baseline's interval is about ±0.034 wide, so with 120 test images a single mean Dice is only known to within
about three hundredths. __ (Which gains over the baseline are significant? Note any model pairs whose difference
is within the noise.) These intervals cover the choice of test images only, not the randomness of training
(Section 8.2).

What the table shows:
- **Baseline.** The U-Net trained from scratch already reaches a Dice of 0.851 and an IoU of 0.774. Its recall
  (0.897) is higher than its precision (0.867): it tends to over-segment a little, marking some healthy tissue
  around the polyp, rather than missing parts of the polyp.
- **Pretrained encoder.** __ (ResNet-34 vs scratch: +__ Dice. Does ImageNet pretraining help with only 800
  training images, and how many epochs did it need compared with 60 for the baseline?)
- **Attention gates.** __ (Attention U-Net vs U-Net: the only difference is the four gates, so this gap is the
  effect of attention. Say whether it is larger than run-to-run noise; with one seed we cannot be sure.)
- **Transformer encoder.** __ (SegFormer vs the best CNN.)
- **Threshold tuning.** For the baseline, the best validation threshold was exactly 0.5, so tuning changed
  nothing. __ (Did any other model gain from a different threshold?)
- **Test-time augmentation** raised the baseline's Dice by 0.012 (0.851 → 0.863), at the cost of four forward
  passes per frame. __ (Same for the other models; the speed cost is discussed in Section 5.)

## 4.3 Training curves
Figure 4.1 (`figures/training_curves.png`) shows the training loss and the validation Dice of every model per
epoch. The baseline reached its best validation Dice (0.855) at epoch 54 of 60 and stayed there (0.852 at the
last epoch), so it converged without overfitting. __ (Pretrained models: how many epochs until the validation
Dice levelled off, compared with the models trained from scratch.)

## 4.4 Qualitative comparison
Figure 4.2 (`figures/model_comparison.png`) shows five test polyps, from the smallest to the largest, with the
ground truth and the mask predicted by each model. __ (Describe what is visible: e.g. which models miss the
smallest polyp, which ones produce cleaner borders on large polyps, any false positives on glare or folds.)

## 4.5 Summary
__ (Two or three sentences: the most accurate model, the best accuracy for its speed, and how big the gap to the
baseline is. Section 6 shows where the remaining errors are.)
