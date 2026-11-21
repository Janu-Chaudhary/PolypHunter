# Abstract

<!-- Owner: Manoj (#6). Final version is written last, once every `__` in sections 4–7 is filled in. About 150 words. -->

Missed polyps during colonoscopy, especially small and flat ones, are a major cause of colorectal cancers that
appear between screenings. We compare five U-Net variants for pixel-level polyp segmentation on Kvasir-SEG
(1000 images; fixed 800/80/120 split): a U-Net trained from scratch, U-Nets with ImageNet-pretrained ResNet-34
and MobileNetV2 encoders, an Attention U-Net and SegFormer with a MiT-B2 transformer encoder. The baseline U-Net
reaches a test Dice of 0.851 and IoU of 0.774; the best model, __, reaches a Dice of __ and IoU of __. Measured
at batch size 1 on a T4 GPU, __ of the five models exceed 25 frames per second, the minimum for live video, and
__ offers the best accuracy for its speed. A failure analysis using expert sessile labels shows that small (< 5%
of the image) and flat polyps remain the hardest: the baseline misses 18.5% of small polyps and none of the
larger ones, and __. On CVC-ClinicDB, from another hospital, Dice drops by __ to __ without retraining.
