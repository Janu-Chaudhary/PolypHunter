# 7. Generalisation to CVC-ClinicDB

<!-- Owner: Manoj (#6). Numbers come from results/summary/generalisation_cvc.csv, written by 04_analysis.
     `__` = waiting for the results of #4 and #5. -->

A model that will be used in another hospital must work on images from other cameras, other lighting and other
patients. To test this, every model trained on Kvasir-SEG was also evaluated on **CVC-ClinicDB** (Bernal et al.,
2015) **without any retraining**: 612 frames from 29 colonoscopy videos recorded in Barcelona with a different
endoscope (Section 2.5). The images are resized and normalised exactly like the Kvasir-SEG images, and the
threshold is 0.5.

**Table 7.1.** Dice on the Kvasir-SEG test set and on CVC-ClinicDB, and the drop between them.

| Model | Kvasir-SEG Dice (n = 120) | CVC-ClinicDB Dice (n = 612) | CVC IoU | Drop |
|---|---|---|---|---|
| U-Net (scratch) | 0.851 | 0.750 | 0.655 | 0.101 |
| U-Net + ResNet-34 | __ | __ | __ | __ |
| U-Net + MobileNetV2 | __ | __ | __ | __ |
| Attention U-Net | __ | __ | __ | __ |
| SegFormer (MiT-B2) | __ | __ | __ | __ |

**Baseline.** The U-Net trained from scratch loses 0.101 Dice on the new hospital's images (0.851 → 0.750).
It still finds most polyps, but a tenth of its overlap is lost, which shows that it has partly learned features
specific to Kvasir-SEG's camera and image style.

__ (For the other models: do the ImageNet-pretrained encoders lose less, as one would expect from features learned
on a million varied photos? Does SegFormer's global self-attention generalise better than convolutions? Rank the
models by CVC Dice, and note whether that ranking differs from the Kvasir ranking.)

**Caveats.**
- CVC-ClinicDB frames come from only 29 videos, so neighbouring frames are very similar and the 612 frames are
  far from 612 independent cases.
- Its images are smaller (384×288) than most Kvasir-SEG images and wider than they are tall. Resizing them to
  352×352 stretches them vertically and changes their apparent sharpness, so polyps look slightly different
  from those the models were trained on.
- We use no CVC image for training or for choosing anything, so these numbers are a fair test of transfer, but a
  single seed means differences of about 0.01 should not be over-interpreted.
