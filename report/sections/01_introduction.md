# 1. Introduction

Colorectal cancer is one of the most common cancers worldwide and one of the leading causes of cancer
death (Sung et al., 2021). Most colorectal cancers grow out of **polyps**, small growths on the inner
wall of the colon that can turn malignant over several years. Finding and removing polyps during a
**colonoscopy**, a video examination of the colon, is therefore the main way to prevent the disease.

Colonoscopy is not perfect: a meta-analysis of tandem colonoscopy studies found that about a quarter of
adenomas are missed (Zhao et al., 2019), and every percentage point gained in a doctor's adenoma detection
rate is linked to a lower risk of cancer appearing before the next screening (Corley et al., 2014). The
polyps missed most often are the **small** ones and the **flat (sessile)** ones, which barely stand out
from the surrounding tissue.

Computer assistance can help here. **Semantic segmentation** labels every pixel of a colonoscopy frame as
polyp or background. Compared with only answering "is there a polyp?", a pixel mask shows the doctor the
polyp's exact location, size and outline, which supports both detection and the decision on how to remove it.
For this to be useful during a live examination, the model must also be fast enough to keep up with the
video stream, typically 25–30 frames per second.

In this project we study **U-Net** (Ronneberger et al., 2015), the standard architecture for biomedical
segmentation, and several of its variants on the public **Kvasir-SEG** dataset (Jha et al., 2020). Each
step changes one idea, so its effect can be measured:

1. a **U-Net trained from scratch**, the baseline;
2. a U-Net whose encoder is **pretrained on ImageNet** (ResNet-34), plus a lightweight **MobileNetV2**
   encoder, to measure the value of transfer learning;
3. an **Attention U-Net** (Oktay et al., 2018), which learns where to look, and **SegFormer**
   (Xie et al., 2021), which replaces the convolutional encoder with a transformer.

**Contributions.**
- A controlled comparison of five U-Net variants on the same fixed split, with the same augmentation,
  loss and evaluation code, reported with Dice and IoU.
- An **accuracy versus inference-speed** study on a single T4 GPU at batch size 1, which shows which models
  can run in real time during a colonoscopy, including the cost of test-time augmentation.
- A **failure analysis on small and flat polyps**, using polyp size, expert sessile/flat labels from
  Kvasir-Sessile (Jha et al., 2021) and a colour-contrast measure, together with miss rates.
- A **cross-dataset test** on CVC-ClinicDB (Bernal et al., 2015), images from another hospital, to show how
  well each model generalises without retraining.
- A public, reproducible code base in which each experiment runs as one notebook on a free GPU.

<!-- References used in this section, for the list in 08_conclusion.md:
H. Sung et al. Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. CA: A Cancer Journal for Clinicians, 2021.
S. Zhao et al. Magnitude, Risk Factors, and Factors Associated With Adenoma Miss Rate of Tandem Colonoscopy: A Systematic Review and Meta-analysis. Gastroenterology, 2019.
D. A. Corley et al. Adenoma Detection Rate and Risk of Colorectal Cancer and Death. New England Journal of Medicine, 2014.
D. Jha et al. A Comprehensive Study on Colorectal Polyp Segmentation with ResUNet++, Conditional Random Field and Test-Time Augmentation (Kvasir-Sessile). IEEE JBHI, 2021.
J. Bernal et al. WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians (CVC-ClinicDB). Computerized Medical Imaging and Graphics, 2015.
-->
