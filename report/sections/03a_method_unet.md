# 3. Method

## 3.1 U-Net baseline and training setup

### Architecture
Our baseline is the U-Net of Ronneberger et al. (2015), an encoder–decoder network shaped like the letter U.

- **Encoder (contracting path).** Five levels, each a *double convolution*: two 3×3 convolutions, each
  followed by batch normalisation and a ReLU. The number of channels doubles at every level:
  64 → 128 → 256 → 512 → 1024. Between levels, a 2×2 max-pooling halves the height and width, so a
  352×352 input becomes 176, 88, 44 and finally **22×22** pixels at the deepest level. Deeper levels see a
  larger part of the image and learn *what* is in it, but they lose *where* exactly it is.
- **Decoder (expanding path).** Four steps. Each step uses a 2×2 transposed convolution to double the
  resolution and halve the channels, then a double convolution, until the output is 352×352 again.
- **Skip connections.** Before each decoder double convolution, the decoder feature map is
  **concatenated with the encoder feature map of the same resolution**. This brings back the fine spatial
  detail (polyp edges) that pooling discarded, and is the key idea of U-Net.
- **Output.** A final 1×1 convolution maps the 64 channels to **one logit per pixel**. A sigmoid turns the
  logit into the probability that the pixel belongs to a polyp; pixels above a threshold (0.5 unless stated
  otherwise) form the predicted mask.

We make two common changes to the original design. First, we add **batch normalisation** after every
convolution, which makes training from scratch faster and more stable. Second, we use **padded convolutions**,
so the output has the same size as the input. The original U-Net used unpadded convolutions and had to crop
the skip connections. Our U-Net has **31.0 million parameters**.

### Loss function
The loss is the sum of two terms:

L = BCE(p, g) + (1 − Dice(p, g))

where *p* are the predicted probabilities and *g* is the ground-truth mask.
- **Binary cross-entropy (BCE)** scores every pixel independently and gives smooth, stable gradients.
- **Soft Dice loss** scores the overlap between the whole predicted polyp and the true polyp. Polyps often
  cover less than 20% of the image, so a model trained with BCE alone can reach a low loss by predicting
  "background" almost everywhere. The Dice term prevents this, because it is computed only from the polyp
  pixels.

### Optimisation
- **AdamW** (learning rate 3·10⁻⁴, weight decay 10⁻⁴), batch size 8.
- **OneCycle learning-rate schedule:** the learning rate rises during the first 10% of the steps (warm-up),
  then decays following a cosine curve, which gives stable early training and a fine final convergence.
- **60 epochs** of 100 steps each (800 images / batch 8).
- **Mixed precision (fp16)** on the GPU, which roughly halves memory use and speeds up training.
  The loss itself is computed in fp32 for numerical stability.
- **Model selection:** after every epoch we compute the Dice score on the validation set and keep the weights
  of the best epoch. The test set plays no part in this choice.
- A fixed random seed (42) controls the weight initialisation and the augmentation, so a rerun gives
  nearly the same model (some GPU operations are not bit-exact between runs).

Training the baseline took 46 minutes on an NVIDIA T4 GPU (60 epochs). The best validation Dice, 0.855,
was reached at epoch 54; by the last epoch it was still 0.852, so the model had converged without
overfitting the validation set.

### Evaluation metrics
For each test image we compare the predicted mask *P* with the ground truth *G* and count true positives
(TP), false positives (FP) and false negatives (FN) pixels:

- **Dice** = 2·TP / (2·TP + FP + FN) = 2|P∩G| / (|P| + |G|)
- **IoU** (Jaccard index) = TP / (TP + FP + FN) = |P∩G| / |P∪G|
- **Precision** = TP / (TP + FP): how much of the prediction is really polyp.
- **Recall** = TP / (TP + FN): how much of the polyp was found.

Each metric is computed **per image and then averaged** over the 120 test images, so a small polyp counts as
much as a large one. Dice is always at least as large as IoU for the same prediction (Dice = 2·IoU / (1 + IoU)).

**Baseline result.** On the test set, the U-Net trained from scratch reaches a **Dice of 0.851** and an
**IoU of 0.774** (precision 0.867, recall 0.897). This is the reference point for every other model in Section 4.
