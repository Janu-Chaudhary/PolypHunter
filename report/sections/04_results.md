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

| Model | Params (M) | Dice @0.5 | IoU @0.5 | Precision | Recall | Threshold | Dice tuned | Dice TTA | FPS (fp16) |
|---|---|---|---|---|---|---|---|---|---|
| U-Net (scratch) | 31.0 | 0.851 | 0.774 | 0.867 | 0.897 | 0.50 | 0.851 | 0.863 | 45.7 |
| U-Net + ResNet-34 | __ | __ | __ | __ | __ | __ | __ | __ | __ |
| U-Net + MobileNetV2 | __ | __ | __ | __ | __ | __ | __ | __ | __ |
| Attention U-Net | __ | __ | __ | __ | __ | __ | __ | __ | __ |
| SegFormer (MiT-B2) | __ | __ | __ | __ | __ | __ | __ | __ | __ |

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

## 4.2 Training curves
Figure 4.1 (`figures/training_curves.png`) shows the training loss and the validation Dice of every model per
epoch. The baseline reached its best validation Dice (0.855) at epoch 54 of 60 and stayed there (0.852 at the
last epoch), so it converged without overfitting. __ (Pretrained models: how many epochs until the validation
Dice levelled off, compared with the models trained from scratch.)

## 4.3 Qualitative comparison
Figure 4.2 (`figures/model_comparison.png`) shows five test polyps, from the smallest to the largest, with the
ground truth and the mask predicted by each model. __ (Describe what is visible: e.g. which models miss the
smallest polyp, which ones produce cleaner borders on large polyps, any false positives on glare or folds.)

## 4.4 Summary
__ (Two or three sentences: the most accurate model, the best accuracy for its speed, and how big the gap to the
baseline is. Section 6 shows where the remaining errors are.)
