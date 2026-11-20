# PolypHunter 🔬
### P13 — Polyp Segmentation in Colonoscopy: U-Net and Its Variants

Pixel-level segmentation of colorectal polyps on **Kvasir-SEG** to support early colorectal-cancer screening.
We go from a U-Net baseline → U-Net with a pretrained ResNet encoder → Attention U-Net / transformer encoder,
measure **Dice** and **IoU**, study the **accuracy vs inference-speed trade-off** for real-time use, and analyse
**failures on small and flat polyps**.

## Models

| Key | Model | Encoder init |
|---|---|---|
| `unet_scratch` | U-Net (Ronneberger et al., 2015) | random |
| `unet_resnet34` | U-Net + ResNet-34 encoder | ImageNet |
| `unet_mobilenetv2` | U-Net + MobileNetV2 encoder (lightweight, real-time reference) | ImageNet |
| `attention_unet` | Attention U-Net (Oktay et al., 2018) | random |
| `segformer_mitb2` | SegFormer, MiT-B2 transformer encoder | ImageNet |

Extras: test-time augmentation (flips), threshold tuned on validation, cross-dataset test on CVC-ClinicDB,
attention-gate visualisation, Kvasir-Sessile labels for flat polyps.

## Results

_Filled in from `results/` after the full Kaggle run._

## Repository layout

```
src/p13_polyp_segmentation.py      all code (percent-format script = the notebook source)
notebooks/p13_polyp_segmentation.ipynb   notebook built from src/ (this is what runs on Kaggle)
tools/make_notebook.py             script -> .ipynb converter
tools/run_on_kaggle.sh             build + push to Kaggle (GPU T4); `smoke` = 1-epoch pipeline check
kaggle/kernel-metadata.json        Kaggle kernel config (GPU, internet, datasets)
results/                           metrics CSV/JSON + figures downloaded from the Kaggle run
report/                            project report
```

## How to reproduce

**On Kaggle (recommended, free T4 GPU):**
1. Open the notebook on Kaggle, attach the datasets `debeshjha1/kvasirseg`, `debeshjha1/kvasirsessile`,
   `balraj98/cvcclinicdb`, set *Accelerator = GPU T4* and *Internet = On*.
2. *Save Version → Save & Run All*. Outputs (CSVs, figures, weights) appear in `/kaggle/working`.

**From the command line:**
```bash
pip install kaggle            # and put your API token in ~/.kaggle/
tools/run_on_kaggle.sh smoke  # quick pipeline check (~5 min)
tools/run_on_kaggle.sh        # full run (~2 h on a T4)
kaggle kernels output <user>/p13-polyp-segmentation -p results
```

Data split: fixed seed 42 → 800 train / 80 validation / 120 test (test size as in Jha et al. 2020).

## Team

| Member | Responsibility |
|---|---|
| Member 1 | Data pipeline, U-Net baseline, metrics, training curves |
| Member 2 | ResNet-34 and MobileNetV2 U-Nets, speed benchmark, TTA trade-off |
| Member 3 | Attention U-Net, SegFormer, attention maps, threshold tuning |
| Member 4 | Failure analysis (small / flat / low-contrast), CVC-ClinicDB generalisation, report assembly |

## References

1. O. Ronneberger, P. Fischer, T. Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015. https://arxiv.org/abs/1505.04597
2. D. Jha et al. *Kvasir-SEG: A Segmented Polyp Dataset.* MMM 2020. https://arxiv.org/abs/1911.07069
3. O. Oktay et al. *Attention U-Net: Learning Where to Look for the Pancreas.* MIDL 2018. https://arxiv.org/abs/1804.03999
4. E. Xie et al. *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS 2021. https://arxiv.org/abs/2105.15203
5. D. Jha et al. *A Comprehensive Study on Colorectal Polyp Segmentation with ResUNet++, Conditional Random Field and Test-Time Augmentation* (Kvasir-Sessile). IEEE JBHI 2021. https://arxiv.org/abs/2107.12435
6. J. Bernal et al. *WM-DOVA maps for accurate polyp highlighting in colonoscopy* (CVC-ClinicDB). CMIG 2015.
