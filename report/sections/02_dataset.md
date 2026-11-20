# 2. Dataset

## 2.1 Kvasir-SEG
Kvasir-SEG (Jha et al., 2020) contains **1000 colonoscopy images of polyps**, each with a pixel-level mask
in which white marks the polyp and black the background. The images were collected at Vestre Viken Health
Trust in Norway. The masks were drawn by a medical doctor and verified by an experienced gastroenterologist.
Image sizes vary from 332×487 to 1920×1072 pixels. Every image contains at least one polyp, and polyps differ
strongly in size, shape, colour and texture, which makes the dataset a realistic test of segmentation models.

Figure 2.1 (`figures/dataset_samples.png`) shows four test images from the smallest to the largest polyp,
with their ground-truth masks. In our test set, the polyp covers between __% and __% of the image
(median __%), and __ of the 120 test images contain a small polyp (under 5% of the image).

## 2.2 Split
We split the 1000 images once with a fixed random seed (42) into **800 training, 80 validation and
120 test images**. The test-set size follows the Kvasir-SEG paper; the split itself is our own. All five
models use exactly the same split, which is stored in `results/split.json`.
- The **training set** is the only data the models learn from.
- The **validation set** is used for every choice made during development: keeping the best epoch
  and choosing the probability threshold.
- The **test set** is used only once per model, for the final numbers. It is never used to choose anything,
  so the reported scores are not optimistically biased.

## 2.3 Preprocessing
Every image is resized to **352×352 pixels**. Images use area interpolation, which keeps detail when
shrinking; masks use nearest-neighbour interpolation so they stay strictly binary. 352 is divisible by 16,
so the four 2×2 poolings of U-Net divide it exactly (352 → 176 → 88 → 44 → 22). Pixel values are scaled to
[0, 1] and normalised with the ImageNet mean and standard deviation. The pretrained encoders were trained on
this normalisation, and we use it for all models so that the input is identical everywhere. The masks are
stored as JPEG files, so we threshold them at 127 to remove compression noise.

## 2.4 Data augmentation
With only 800 training images, the models would quickly memorise the training set. During training we
therefore apply random augmentations, identically to the image and its mask, every time an image is loaded:

| Augmentation | Setting | Why |
|---|---|---|
| Horizontal and vertical flip | each with probability 0.5 | a colonoscope has no fixed "up" |
| Rotation by a multiple of 90° | uniform over 0°, 90°, 180°, 270° | same reason, and it is lossless |
| Rotation, scale and shift | probability 0.7: ±25°, scale 0.85–1.15, shift up to 6% | changing camera angle and distance |
| Brightness and contrast | probability 0.6: contrast ×0.8–1.2, brightness ±25 | changing light and exposure |

Validation and test images are never augmented.

## 2.5 Additional evaluation data
Two further datasets are used **only for evaluation**, never for training:
- **Kvasir-Sessile** (Jha et al., 2021): 196 of the Kvasir-SEG polyps that experts labelled as **sessile
  (flat) and smaller than 10 mm**. We use it to tag the flat polyps in our test set (__ of the 120 test
  images) for the failure analysis.
- **CVC-ClinicDB** (Bernal et al., 2015): 612 frames (384×288 pixels) from 29 colonoscopy videos recorded at
  Hospital Clínic in Barcelona, with expert masks. Because it comes from another hospital and another
  camera, it tests how well each model generalises beyond Kvasir-SEG.
