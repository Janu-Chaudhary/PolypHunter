# Abstract

<!-- Owner: Manoj (#6). About 150 words. Numbers match Sections 4–8. -->

Polyps missed during colonoscopy, most often small and flat ones, are linked to colorectal cancers that appear
between screenings. We compare five U-Net variants for polyp segmentation on Kvasir-SEG (1000 images, fixed
800/80/120 split): a U-Net trained from scratch, U-Nets with ImageNet-pretrained ResNet-34 and MobileNetV2
encoders, an Attention U-Net and SegFormer (MiT-B2). The baseline reaches a test Dice of 0.851. Attention gates
add nothing significant; pretrained encoders do: ResNet-34 reaches 0.903 and SegFormer 0.917, a difference that is
not significant under a paired bootstrap test. All five models exceed the 25 frames per second of live video on a
T4 GPU, ResNet-34 the most (110). Using expert sessile labels, we find that small and flat polyps remain the
hardest: the baseline misses 18.5% of small polyps, ResNet-34 and SegFormer 7.4%. On CVC-ClinicDB, from another
hospital, SegFormer loses only 0.05 Dice without retraining. We recommend SegFormer where accuracy matters most and
ResNet-34 where speed does.
