# 6. Failure analysis: small and flat polyps

<!-- Owner: Manoj (#6). Numbers come from results/summary/failure_by_*.csv, written by 04_analysis.
     Baseline numbers are real; `__` = waiting for the results of #4 and #5. -->

A mean Dice of 0.85 hides where a model fails. Clinically, the polyps that doctors miss most often are the
**small** and the **flat (sessile)** ones (Section 1). In this section we split the 120 test images into groups
and check whether the models fail on the same polyps that doctors do.

## 6.1 How the groups are defined
- **Size:** the polyp's share of the image, measured on the original-resolution mask.
  Small < 5%, medium 5–20%, large > 20%.
- **Morphology:** a test image is **sessile/flat** if it is listed in Kvasir-Sessile (Jha et al., 2021), a set of
  196 Kvasir-SEG polyps that experts labelled sessile and smaller than 10 mm. This is a real expert label.
- **Contrast:** the colour difference (ΔE in CIELAB colour space) between the polyp and a ring of tissue
  12 pixels wide around it, measured on the 352×352 image. The test set is split into thirds: low, mid and high contrast. A low value means the
  polyp blends into the colon wall. This is only a **stand-in** for "hard to see", computed by us, not an expert label.
- **Miss rate:** the share of images with Dice < 0.5, meaning the polyp was essentially not found. A missed polyp
  is clinically worse than a slightly wrong outline, so we report it next to the mean Dice.

Some groups are small (n = 27 small polyps, 31 sessile polyps), so a single image moves a group's mean noticeably.
We therefore give **n** for every group and treat differences of a few hundredths with caution.

## 6.2 Results by group
**Table 6.1.** Mean test Dice (miss rate) per group.

| Group | n | U-Net (scratch) | ResNet-34 | MobileNetV2 | Attention U-Net | SegFormer |
|---|---|---|---|---|---|---|
| Small (< 5%) | 27 | 0.713 (18.5%) | __ | __ | __ | __ |
| Medium (5–20%) | 58 | 0.900 (0.0%) | __ | __ | __ | __ |
| Large (> 20%) | 35 | 0.876 (0.0%) | __ | __ | __ | __ |
| Sessile/flat | 31 | 0.734 (12.9%) | __ | __ | __ | __ |
| Other | 89 | 0.892 (1.1%) | __ | __ | __ | __ |
| Low contrast | 40 | 0.836 (5.0%) | __ | __ | __ | __ |
| Mid contrast | 40 | 0.825 (7.5%) | __ | __ | __ | __ |
| High contrast | 40 | 0.892 (0.0%) | __ | __ | __ | __ |

Figure 6.1 (`figures/failure_analysis.png`) shows the same means as bar charts, and Figure 6.2
(`figures/dice_vs_size.png`) plots the Dice of every test image against its polyp size.

**Small polyps are the main weakness.** The baseline scores 0.713 on small polyps against 0.900 on medium ones,
and it misses 5 of the 27 small polyps (18.5%). It misses no medium or large polyp at all. All five complete
misses in the test set are small polyps. __ (Do the pretrained and transformer models close this gap, or only
raise the score on polyps that were already easy?)

**Flat polyps are almost as hard.** Sessile polyps score 0.734 against 0.892 for the others, with a miss rate of
12.9% against 1.1%. __ (Same comparison for the other models.)

**Small and flat overlap.** The two groups are largely the same polyps: 18 of the 27 small polyps are sessile,
because Kvasir-Sessile only contains polyps under 10 mm. Separating the two factors for the baseline:

**Table 6.2.** Baseline Dice by size and morphology together (n in brackets).

|  | Not small | Small (< 5%) |
|---|---|---|
| **Other** | 0.899 (80) | 0.826 (9) |
| **Sessile/flat** | 0.840 (13) | 0.657 (18) |

Both factors lower the Dice on their own, and polyps that are both small and flat are by far the hardest
(0.657); the baseline misses 4 of these 18 polyps (22%). With only 9 and 13 images in two of the cells, this split is indicative only.

**Contrast matters less than size.** Low- and mid-contrast polyps score about 0.83, high-contrast ones 0.89.
The effect is smaller than for size, and low and mid contrast are not clearly different. Sessile polyps do have
lower contrast (median ΔE 19.6 against 42.0), which fits the clinical picture of flat lesions that blend into the
wall. __ (Check the trend for the other models.)

## 6.3 Worst cases
Figure 6.3 (`figures/worst_cases.png`) shows the eight test images on which the best model, __, scored lowest,
with the ground truth in green and the prediction in red. Likely reasons:

1. __ (image name, area, Dice): __ (e.g. very small polyp, polyp at the image edge, blur, glare, polyp blending
   into the wall, a fold mistaken for a polyp).
2. __
3. __
4. __
5. __
6. __
7. __
8. __

__ (One sentence on the common pattern, e.g. "six of the eight are small sessile polyps".)

## 6.4 What this means
__ (Which model is most reliable on small and flat polyps, the ones that matter most clinically? Is its miss rate
low enough to be useful, and which kind of polyp would still need the doctor's full attention?)
