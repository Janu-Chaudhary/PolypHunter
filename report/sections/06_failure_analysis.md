# 6. Failure analysis: small and flat polyps

<!-- Owner: Manoj (#6). Numbers come from results/summary/failure_by_*.csv, written by 04_analysis.
     `__` = waiting for the Kvasir-SEG images (worst-case gallery). -->

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
| Small (< 5%) | 27 | 0.713 (18.5%) | 0.851 (7.4%) | 0.812 (11.1%) | 0.730 (22.2%) | 0.862 (7.4%) |
| Medium (5–20%) | 58 | 0.900 (0.0%) | 0.934 (0.0%) | 0.934 (1.7%) | 0.906 (0.0%) | 0.948 (0.0%) |
| Large (> 20%) | 35 | 0.876 (0.0%) | 0.891 (5.7%) | 0.890 (5.7%) | 0.862 (0.0%) | 0.908 (2.9%) |
| Sessile/flat | 31 | 0.734 (12.9%) | 0.850 (6.5%) | 0.826 (9.7%) | 0.758 (16.1%) | 0.867 (6.5%) |
| Other | 89 | 0.892 (1.1%) | 0.922 (2.2%) | 0.917 (3.4%) | 0.887 (1.1%) | 0.934 (1.1%) |
| Low contrast | 40 | 0.836 (5.0%) | 0.896 (2.5%) | 0.883 (7.5%) | 0.861 (2.5%) | 0.907 (2.5%) |
| Mid contrast | 40 | 0.825 (7.5%) | 0.869 (7.5%) | 0.865 (7.5%) | 0.823 (10.0%) | 0.898 (5.0%) |
| High contrast | 40 | 0.892 (0.0%) | 0.943 (0.0%) | 0.933 (0.0%) | 0.876 (2.5%) | 0.946 (0.0%) |

Figure 6.1 (`figures/failure_analysis.png`) shows the same means as bar charts, and Figure 6.2
(`figures/dice_vs_size.png`) plots the Dice of every test image against its polyp size.

**Small polyps are the main weakness.** The baseline scores 0.713 on small polyps against 0.900 on medium ones,
and it misses 5 of the 27 small polyps (18.5%). It misses no medium or large polyp at all: all five complete
misses in the test set are small polyps. The Attention U-Net behaves the same way (0.730 on small polyps, 6 of
27 missed, again all of its misses are small polyps), so the attention gates do not help where help is needed
most. **Pretrained encoders narrow the gap most.** ResNet-34 raises the Dice on small polyps by 0.138
(0.713 → 0.851) and SegFormer by 0.149 (→ 0.862), far more than their gains on medium polyps (+0.034 and
+0.048). Both miss only 2 of the 27 small polyps (7.4%); MobileNetV2 is in between (0.812, 3 missed). The
improvement from a pretrained encoder is therefore largest exactly on the hard cases, not only on polyps that
were already easy.

**Flat polyps are almost as hard.** For the baseline, sessile polyps score 0.734 against 0.892 for the others,
with a miss rate of 12.9% against 1.1%. The Attention U-Net is similar (0.758, 16.1% missed). ResNet-34 and
SegFormer halve the miss rate on sessile polyps, to 6.5% (2 of 31), with Dice 0.850 and 0.867; MobileNetV2
lowers it less, to 9.7% (Dice 0.826). A gap to the other polyps remains for every model
(SegFormer 0.867 against 0.934).

**Small and flat overlap.** The two groups are largely the same polyps: 18 of the 27 small polyps are sessile,
because Kvasir-Sessile only contains polyps under 10 mm. Table 6.2 separates the two factors.

**Table 6.2.** Mean test Dice (miss rate) by size and morphology together.

| Group | n | U-Net (scratch) | ResNet-34 | MobileNetV2 | Attention U-Net | SegFormer |
|---|---|---|---|---|---|---|
| Other, not small | 80 | 0.899 (0.0%) | 0.927 (1.2%) | 0.924 (2.5%) | 0.896 (0.0%) | 0.942 (0.0%) |
| Other, small | 9 | 0.826 (11.1%) | 0.874 (11.1%) | 0.858 (11.1%) | 0.804 (11.1%) | 0.870 (11.1%) |
| Sessile, not small | 13 | 0.840 (0.0%) | 0.863 (7.7%) | 0.878 (7.7%) | 0.848 (0.0%) | 0.880 (7.7%) |
| Sessile, small | 18 | 0.657 (22.2%) | 0.840 (5.6%) | 0.789 (11.1%) | 0.693 (27.8%) | 0.858 (5.6%) |

For the two CNNs trained from scratch, both factors lower the Dice on their own, and polyps that are both small
and flat are by far the hardest: the baseline misses 4 of these 18 polyps and the Attention U-Net 5. The
pretrained encoders shrink this weakness: MobileNetV2 misses 2 of the 18, and ResNet-34 and SegFormer only 1
each. For SegFormer (0.858) and ResNet-34 (0.840), the Dice on small sessile polyps is close to their Dice on the
other groups; MobileNetV2 still lags (0.789). With only 9 and 13 images in two of the cells, where one image
changes the miss rate by 8–11 points, this split is indicative only.

**Contrast matters less than size.** All five models score lower on low- and mid-contrast polyps than on
high-contrast ones: for the baseline about 0.83 against 0.89, for ResNet-34 0.896 and 0.869 against 0.943, and
for SegFormer 0.907 and 0.898 against 0.946. The effect is smaller than for size, and low and mid contrast are
not clearly different. Sessile polyps do have lower contrast (median ΔE 19.6 against 42.0), which fits the
clinical picture of flat lesions that blend into the wall.

**A new failure mode: large polyps.** The models trained from scratch miss no polyp larger than 5% of the image,
but the three pretrained models each miss one or two large ones. Two large polyps (28% and 31% of the image, the
second one sessile) are missed by both ResNet-34 and MobileNetV2, and the second one also by SegFormer, while
both scratch models find them. Pretraining therefore trades many misses on small polyps for a few misses on
large ones. __ (Check these two images in the worst-case gallery: e.g. an unusual colour, a polyp filling most of
the frame, or glare.)

## 6.3 Worst cases
Figure 6.3 (`figures/worst_cases.png`) shows the eight test images on which the best model, SegFormer, scored
lowest, with the ground truth in green and the prediction in red. Likely reasons:

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
SegFormer and U-Net + ResNet-34 are the most reliable models on the polyps that matter most clinically. Each
misses only 2 of the 27 small polyps and 1 of the 18 small sessile polyps, against 5 and 4 for the baseline, and
they miss 3 and 4 of the 120 test polyps overall, against 5 for the baseline. Their misses are not limited to
small polyps, as the scratch models' are: each misses at least one large polyp, so they fail less often but not
in a fully predictable way. Small and flat polyps remain the weakest groups for every model, so they still need
the doctor's full attention.
