## 3.2 Pretrained encoders: ResNet-34 and MobileNetV2

### Why pretrain the encoder
Kvasir-SEG has only 800 training images, far too few to teach a U-Net encoder to recognise edges,
textures and shapes from scratch (Section 3.1's `unet_scratch` has to learn all of that from polyp images
alone). Transfer learning swaps the encoder for one already trained on ImageNet's 1.2 million natural
images: it starts out already knowing general-purpose visual features, and only has to adapt them to
colonoscopy images and learn the decoder from zero. This is the standard fix for small medical-imaging
datasets, where collecting and labelling more data is expensive. Both encoders below come from
`segmentation_models_pytorch`, which attaches a U-Net decoder (same skip-connection design as Section 3.1)
to a pretrained backbone.

### ResNet-34 encoder (`unet_resnet34`)
Built with `smp.Unet("resnet34", encoder_weights="imagenet")`. ResNet-34's residual blocks add the input
of a block back onto its output (`y = F(x) + x`) instead of only passing it through convolutions. This
shortcut lets gradients flow directly through the skip path during backpropagation, which is what makes
very deep encoders like this one trainable at all, and is also why its ImageNet features transfer well: the
network can reuse early, general-purpose filters almost unchanged while later layers specialise. At
**24.4 M parameters** it is smaller than the from-scratch baseline (31.0 M, Section 3.1) despite being a
deeper network, because standard residual blocks use fewer, wider convolutions than our baseline's
double-convolution blocks.

### MobileNetV2 encoder (`unet_mobilenetv2`)
Built with `smp.Unet("mobilenet_v2", encoder_weights="imagenet")`. MobileNetV2 replaces each standard 3×3
convolution with a **depthwise-separable** block: a 3×3 convolution applied one channel at a time
(depthwise), followed by a 1×1 convolution that mixes channels (pointwise). A standard 3×3 convolution over
`C` channels costs `9·C²` multiply-adds per pixel; the depthwise-separable version costs `9·C + C²`, roughly
**8-9× fewer operations** for the channel counts used here. This is the encoder that trades some accuracy for
speed, at only **6.6 M parameters** — about a quarter of ResNet-34's size and a fifth of the baseline's. It
is our candidate for a model light enough to run in real time (Section 5).

### Training setup
Both models train for **40 epochs**, fewer than the baseline's 60, because a pretrained encoder already
starts close to a good solution and converges faster — most of the remaining learning is in the decoder and
in adapting the encoder to this domain, not in re-learning low-level features. The learning rate is lower
than the baseline's 3·10⁻⁴: **2·10⁻⁴ for ResNet-34** and **3·10⁻⁴ for MobileNetV2**. A smaller rate keeps the
early optimisation steps from making large updates to the pretrained weights before the new decoder has
learned anything useful — large early updates would otherwise erase the ImageNet features before they can
help. Loss, optimiser (AdamW), schedule (OneCycle), batch size and evaluation metrics are unchanged from
Section 3.1.

On the test set, `unet_resnet34` reaches a **Dice of 0.903** (IoU 0.846) and `unet_mobilenetv2` reaches a
**Dice of 0.894** (IoU 0.835) — both well above the from-scratch baseline's 0.851, confirming that ImageNet
pretraining helps substantially on this small dataset even though the source images (natural photos) look
nothing like colonoscopy frames. The full comparison across all five models is in Section 4.
