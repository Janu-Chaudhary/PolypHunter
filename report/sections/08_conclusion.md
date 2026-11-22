# 8. Conclusion and limitations

<!-- Owner: Manoj (#6). The recommendation matches Section 5 (Trivedh). -->

## 8.1 Conclusion
We compared five U-Net variants for polyp segmentation on Kvasir-SEG, all trained and evaluated with the same
split, augmentation, loss and code. The U-Net trained from scratch is already a strong baseline (Dice 0.851,
IoU 0.774). Adding attention gates changed nothing measurable (+0.003 Dice, p = 0.73). What mattered was a
pretrained encoder: ImageNet-pretrained CNN encoders raised the Dice significantly (ResNet-34 +0.052 to 0.903,
MobileNetV2 +0.043 to 0.894), and SegFormer's pretrained transformer encoder gave the largest gain (+0.066, to
0.917). SegFormer's lead over ResNet-34 (+0.014) is not statistically significant on 120 test images (p = 0.17).

**Which model to deploy.** All five models run above the 25 frames per second of live colonoscopy video on a T4
GPU at batch size 1 (in fp16). The choice depends on what matters most:
- **Where accuracy matters most: SegFormer (MiT-B2).** It is the most accurate model (Dice 0.917, 0.926 with
  test-time augmentation), misses the fewest polyps (3 of 120) and transfers best to another hospital. It runs at
  56.9 frames per second in fp32, its faster precision, and still at 32.8 with test-time augmentation.
- **Where speed matters most: U-Net + ResNet-34.** It is the fastest model (109.5 frames per second in fp16, 48.7
  with test-time augmentation), its accuracy is within the error margin of SegFormer's, it misses as few small
  polyps (2 of 27), and it is the second most robust model on CVC-ClinicDB.

We do not recommend U-Net + MobileNetV2 despite its small size (6.6 M parameters): it is not faster than
ResNet-34 in fp16, it is significantly less accurate than SegFormer (p = 0.005), it misses more polyps (6 of 120)
and it loses the most accuracy on images from another hospital.

**Where models still fail.** Small polyps (under 5% of the image) and flat (sessile) polyps are the hardest cases
for every model we trained, and these are exactly the polyps that doctors miss most often. For both models trained
from scratch, every complete miss in the test set was a small polyp, most of them also sessile. Pretrained encoders
reduce these failures most: ResNet-34 and SegFormer each miss 1 of the 18 small sessile polyps, against 4 for the
baseline. They instead miss one or two large polyps that the scratch models find. A segmentation model can support
a doctor, but it does not yet remove the need for a careful look at small, flat lesions.

**Generalisation.** On CVC-ClinicDB, images from another hospital, the models trained from scratch lost about
0.1 Dice without retraining (baseline 0.101, Attention U-Net 0.113). SegFormer lost only 0.054 and still reached
0.863, ResNet-34 lost 0.075, but MobileNetV2 lost 0.141, the most of any model. ImageNet pretraining helps a model
transfer to new equipment, but it does not guarantee it.

## 8.2 Limitations
- **Small test set.** 120 test images give reasonably stable overall means, but the failure-analysis groups hold
  only 9–89 images, so differences of a few hundredths between models or groups may be noise.
- **One random seed.** Every model was trained once. Our bootstrap intervals (Section 4.2) show how much the
  results depend on which images are in the test set, but not how much they would change if the same model were
  trained again with another seed; small differences between models could change with another seed.
- **Contrast is a stand-in.** Our "low contrast" group comes from a colour measure we computed, not from expert
  labels. Only the sessile/flat label comes from experts (Kvasir-Sessile).
- **One resolution, one dataset for training.** All models see 352×352 images, which may erase the smallest polyps;
  a higher resolution might help them. All models were trained on one hospital's images.
- **Still images, not video.** We segment single frames. A real system would see a video stream, where the same
  polyp appears in many consecutive frames, and speed was measured on a T4 GPU rather than on hospital hardware.
  Each model's speed was measured once, in its own Kaggle session, so small speed differences may be noise.

## 8.3 Future work
- Train at a higher resolution, or crop around candidate regions, to help with small polyps.
- Repeat every run with several seeds and report the mean and spread, so the intervals also cover training randomness.
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
10. B. Efron, R. J. Tibshirani. An Introduction to the Bootstrap. Chapman & Hall/CRC, 1993.
