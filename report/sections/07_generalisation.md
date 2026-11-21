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
| Attention U-Net | 0.854 | 0.741 | 0.646 | 0.113 |
| SegFormer (MiT-B2) | 0.917 | 0.863 | 0.787 | 0.054 |

**Models trained from scratch.** The baseline loses 0.101 Dice on the new hospital's images (0.851 → 0.750). It
still finds most polyps, but a tenth of its overlap is lost, which shows that it has partly learned features
specific to Kvasir-SEG's camera and image style. The Attention U-Net loses slightly more (0.113) and ends up a
little below the baseline on CVC (0.741 against 0.750): the attention gates do not make the model more robust
to a new hospital either.

**SegFormer.** SegFormer loses only 0.054 Dice, about half as much as the two models trained from scratch, and
keeps a Dice of 0.863 on CVC-ClinicDB. Its lead over the baseline grows from 0.066 on Kvasir-SEG to 0.113 on
CVC-ClinicDB. Two of its properties may explain this, though our experiment cannot separate them: its encoder was
pretrained on ImageNet, so it starts from features learned on over a million varied photographs rather than on
800 colonoscopy images; and its self-attention relates every part of the image to every other part, which may make
it rely less on local colour and texture that change between cameras.
__ (ResNet-34 and MobileNetV2 are also ImageNet-pretrained but are CNNs: if they also lose little, pretraining is
the main reason; if they lose as much as the baseline, the transformer encoder is.)

The ranking on CVC-ClinicDB is the same as on Kvasir-SEG for these three models. __ (Check with all five.)

**Caveats.**
- CVC-ClinicDB frames come from only 29 videos, so neighbouring frames are very similar and the 612 frames are
  far from 612 independent cases.
- Its images are smaller (384×288) than most Kvasir-SEG images and wider than they are tall. Resizing them to
  352×352 stretches them vertically and changes their apparent sharpness, so polyps look slightly different
  from those the models were trained on.
- We use no CVC image for training or for choosing anything, so these numbers are a fair test of transfer, but a
  single seed means differences of about 0.01 should not be over-interpreted.
