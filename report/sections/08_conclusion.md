# 8. Conclusion and limitations

<!-- Owner: Manoj (#6). `__` = waiting for the results of #4 and #5. -->

## 8.1 Conclusion
We compared five U-Net variants for polyp segmentation on Kvasir-SEG, all trained and evaluated with the same
split, augmentation, loss and code. The U-Net trained from scratch is already a strong baseline (Dice 0.851,
IoU 0.774). __ (What each step of the brief added: pretrained encoder +__, attention gates +__, transformer
encoder +__.)

**Which model to deploy.** __ (Recommend one model and justify it with three numbers: its Dice, its frames per
second at batch 1 compared with the 25 FPS needed for live video, and its miss rate on small polyps. If the most
accurate model is too slow, say which model is the best compromise.)

**Where models still fail.** Small polyps (under 5% of the image) and flat (sessile) polyps are the hardest cases
for every model we trained, and these are exactly the polyps that doctors miss most often. For the baseline, all
complete misses in the test set were small polyps, most of them also sessile. __ (Did better models reduce the
miss rate on these polyps?) A segmentation model can support a doctor, but it does not yet remove the need for a
careful look at small, flat lesions.

**Generalisation.** On CVC-ClinicDB, images from another hospital, the baseline lost 0.101 Dice without retraining.
__ (Which model transferred best.)

## 8.2 Limitations
- **Small test set.** 120 test images give reasonably stable overall means, but the failure-analysis groups hold
  only 9–89 images, so differences of a few hundredths between models or groups may be noise.
- **One random seed.** Every model was trained once. Differences between models smaller than about 0.01 Dice
  could change with another seed, and we report no confidence intervals.
- **Contrast is a stand-in.** Our "low contrast" group comes from a colour measure we computed, not from expert
  labels. Only the sessile/flat label comes from experts (Kvasir-Sessile).
- **One resolution, one dataset for training.** All models see 352×352 images, which may erase the smallest polyps;
  a higher resolution might help them. All models were trained on one hospital's images.
- **Still images, not video.** We segment single frames. A real system would see a video stream, where the same
  polyp appears in many consecutive frames, and speed was measured on a T4 GPU rather than on hospital hardware.

## 8.3 Future work
- Train at a higher resolution, or crop around candidate regions, to help with small polyps.
- Repeat every run with several seeds and report the mean and spread.
- Add CVC-ClinicDB or other centres to the training data, and evaluate on video sequences.

## References

1. O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015. https://arxiv.org/abs/1505.04597
2. D. Jha et al. Kvasir-SEG: A Segmented Polyp Dataset. MMM 2020. https://arxiv.org/abs/1911.07069
3. O. Oktay et al. Attention U-Net: Learning Where to Look for the Pancreas. MIDL 2018. https://arxiv.org/abs/1804.03999
4. E. Xie et al. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS 2021. https://arxiv.org/abs/2105.15203
5. D. Jha et al. A Comprehensive Study on Colorectal Polyp Segmentation with ResUNet++, Conditional Random Field and Test-Time Augmentation (Kvasir-Sessile). IEEE JBHI 2021. https://arxiv.org/abs/2107.12435
6. J. Bernal et al. WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians (CVC-ClinicDB). Computerized Medical Imaging and Graphics, 2015.
7. H. Sung et al. Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. CA: A Cancer Journal for Clinicians, 2021.
8. S. Zhao et al. Magnitude, Risk Factors, and Factors Associated With Adenoma Miss Rate of Tandem Colonoscopy: A Systematic Review and Meta-analysis. Gastroenterology, 2019.
9. D. A. Corley et al. Adenoma Detection Rate and Risk of Colorectal Cancer and Death. New England Journal of Medicine, 2014.
