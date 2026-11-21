# 6. Failure analysis: small and flat polyps

<!-- Owner: Manoj (#6). Numbers come from results/summary/failure_by_*.csv, written by 04_analysis.
     `__` = waiting for the results of #4 (Trivedh). -->

A mean Dice of 0.85 or 0.92 hides where a model fails. Clinically, the polyps that doctors miss most often are the
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
| Small (< 5%) | 27 | 0.713 (18.5%) | __ | __ | 0.730 (22.2%) | 0.862 (7.4%) |
| Medium (5–20%) | 58 | 0.900 (0.0%) | __ | __ | 0.906 (0.0%) | 0.948 (0.0%) |
| Large (> 20%) | 35 | 0.876 (0.0%) | __ | __ | 0.862 (0.0%) | 0.908 (2.9%) |
| Sessile/flat | 31 | 0.734 (12.9%) | __ | __ | 0.758 (16.1%) | 0.867 (6.5%) |
| Other | 89 | 0.892 (1.1%) | __ | __ | 0.887 (1.1%) | 0.934 (1.1%) |
| Low contrast | 40 | 0.836 (5.0%) | __ | __ | 0.861 (2.5%) | 0.907 (2.5%) |
| Mid contrast | 40 | 0.825 (7.5%) | __ | __ | 0.823 (10.0%) | 0.898 (5.0%) |
| High contrast | 40 | 0.892 (0.0%) | __ | __ | 0.876 (2.5%) | 0.946 (0.0%) |

Figure 6.1 (`figures/failure_analysis.png`) shows the same means as bar charts, and Figure 6.2
(`figures/dice_vs_size.png`) plots the Dice of every test image against its polyp size.

**Small polyps are the main weakness.** The baseline scores 0.713 on small polyps against 0.900 on medium ones,
and it misses 5 of the 27 small polyps (18.5%). It misses no medium or large polyp at all: all five complete
misses in the test set are small polyps. The Attention U-Net behaves the same way (0.730 on small polyps, 6 of
27 missed, again all of its misses are small polyps), so the attention gates do not help where help is needed
most. **SegFormer narrows the gap most:** it raises the Dice on small polyps by 0.149 (0.713 → 0.862), far more
than its gain on medium polyps (0.900 → 0.948), and misses only 2 of the 27 (7.4%). Its improvement is therefore
largest exactly on the hard cases, not only on polyps that were already easy. __ (ResNet-34 and MobileNetV2.)

**Flat polyps are almost as hard.** For the baseline, sessile polyps score 0.734 against 0.892 for the others,
with a miss rate of 12.9% against 1.1%. The Attention U-Net is similar (0.758, 16.1% missed). SegFormer reaches
0.867 on sessile polyps and halves the miss rate (6.5%), but a gap to the other polyps remains (0.934).
__ (ResNet-34 and MobileNetV2.)

**Small and flat overlap.** The two groups are largely the same polyps: 18 of the 27 small polyps are sessile,
because Kvasir-Sessile only contains polyps under 10 mm. Table 6.2 separates the two factors.

**Table 6.2.** Mean test Dice (miss rate) by size and morphology together.

| Group | n | U-Net (scratch) | ResNet-34 | MobileNetV2 | Attention U-Net | SegFormer |
|---|---|---|---|---|---|---|
| Other, not small | 80 | 0.899 (0.0%) | __ | __ | 0.896 (0.0%) | 0.942 (0.0%) |
| Other, small | 9 | 0.826 (11.1%) | __ | __ | 0.804 (11.1%) | 0.870 (11.1%) |
| Sessile, not small | 13 | 0.840 (0.0%) | __ | __ | 0.848 (0.0%) | 0.880 (7.7%) |
| Sessile, small | 18 | 0.657 (22.2%) | __ | __ | 0.693 (27.8%) | 0.858 (5.6%) |

For the two CNNs trained from scratch, both factors lower the Dice on their own, and polyps that are both small
and flat are by far the hardest: the baseline misses 4 of these 18 polyps and the Attention U-Net 5. SegFormer
is the exception. Its Dice on small sessile polyps (0.858) is close to its Dice on the other groups, and it
misses only 1 of the 18. With only 9 and 13 images in two of the cells, where one image changes the miss rate
by 8–11 points, this split is indicative only.

**Contrast matters less than size.** For the baseline, low- and mid-contrast polyps score about 0.83 and
high-contrast ones 0.89; SegFormer shows the same pattern at a higher level (0.907 and 0.898 against 0.946). The
effect is smaller than for size, and low and mid contrast are not clearly different. Sessile polyps do have
lower contrast (median ΔE 19.6 against 42.0), which fits the clinical picture of flat lesions that blend into the
wall.

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
Of the models trained so far, SegFormer is the most reliable on the polyps that matter most clinically: it misses
3 of the 120 test polyps, against 5 for the baseline, and only 1 of the 18 small sessile polyps. Its misses are not
limited to small polyps (one of the three is a large sessile polyp), so it fails less often but not in a fully
predictable way. Small and flat polyps remain its weakest groups, so they still need the doctor's full attention.
__ (ResNet-34 and MobileNetV2; confirm that the most reliable model is still SegFormer.)
