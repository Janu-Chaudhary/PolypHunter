"""
Write fake results in the team's output format to results/_fake/, so the analysis notebook (04) can be
built before any real training run has finished.

The numbers are invented but shaped like real ones (pretrained models score higher, small polyps score
lower, probability maps are blobs). Never use them in the report. Run:  python3 tools/make_fake_results.py
"""
import json, os
import numpy as np, pandas as pd

OUT = os.path.join("results", "_fake")
MODELS = {  # model -> (mean test Dice, params in millions, fp16 FPS on a T4): rough, invented values
    "unet_scratch": (0.78, 31.0, 55), "unet_resnet34": (0.88, 24.4, 115), "unet_mobilenetv2": (0.85, 6.6, 140),
    "attention_unet": (0.80, 31.4, 50), "segformer_mitb2": (0.90, 24.7, 45)}
N_TEST, SIZE = 120, 352                               # same test size and prediction size as the real runs
rng = np.random.default_rng(0)


def blob(cx, cy, r):
    """A filled circle of radius r at (cx, cy) on a SIZE x SIZE grid, as a uint8 probability map (0 or 255)."""
    yy, xx = np.mgrid[:SIZE, :SIZE]
    return (((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2).astype(np.uint8) * 255


def main():
    """Write split.json, test_attributes.csv and one results/_fake/<model>/ folder per model."""
    os.makedirs(OUT, exist_ok=True)
    names = [f"fake_{i:03d}" for i in range(N_TEST)]
    area = rng.lognormal(np.log(0.10), 0.9, N_TEST).clip(0.005, 0.6)      # polyp share of the image, 0.5%-60%
    attrs = pd.DataFrame({"name": names, "area_frac": area, "contrast": rng.gamma(4, 6, N_TEST),
                          "sessile": (rng.random(N_TEST) < 0.2).astype(int)})          # ~20% flat, like Kvasir
    attrs.to_csv(os.path.join(OUT, "test_attributes.csv"), index=False)
    json.dump({"train": [], "val": [], "test": names}, open(os.path.join(OUT, "split.json"), "w"), indent=1)

    for m, (mu, params, f16) in MODELS.items():
        d = os.path.join(OUT, m); os.makedirs(d, exist_ok=True)
        # Small and flat polyps are made harder so the failure-analysis plots show a visible effect.
        dice = np.clip(mu - 0.25 * (area < 0.05) - 0.1 * attrs.sessile + rng.normal(0, 0.08, N_TEST), 0, 1)
        iou = dice / (2 - dice)                                           # exact Dice -> IoU relation
        pd.DataFrame({"name": names, "dice": dice, "iou": iou, "precision": np.clip(dice + 0.03, 0, 1),
                      "recall": np.clip(dice - 0.02, 0, 1)}).to_csv(os.path.join(d, "per_image_test.csv"), index=False)
        ep = 60 if m in ("unet_scratch", "attention_unet") else 40
        e = np.arange(1, ep + 1)
        pd.DataFrame({"epoch": e, "train_loss": 1.2 * np.exp(-e / 12) + 0.15,
                      "val_dice": mu * (1 - np.exp(-e / 8)), "val_iou": mu * 0.85 * (1 - np.exp(-e / 8))}
                     ).to_csv(os.path.join(d, "history.csv"), index=False)
        thr = {str(t): float(mu - 0.4 * (t - 0.45) ** 2) for t in np.round(np.arange(0.20, 0.81, 0.05), 2)}
        json.dump({"model": m, "dice": float(dice.mean()), "iou": float(iou.mean()), "precision": float(dice.mean() + 0.03),
                   "recall": float(dice.mean() - 0.02), "thr_val": 0.45, "dice_tuned": float(dice.mean() + 0.005),
                   "iou_tuned": float(iou.mean() + 0.005), "val_dice_by_threshold": thr,
                   "dice_tta": float(dice.mean() + 0.01), "iou_tta": float(iou.mean() + 0.01),
                   "cvc_dice": float(dice.mean() - 0.12), "cvc_iou": float(iou.mean() - 0.12),
                   "params_M": params, "fps_fp32": f16 / 2.5, "fps_fp16": float(f16), "fps_fp16_tta": f16 / 3.5,
                   "train_min": ep * 0.5, "epochs": ep, "lr": 3e-4, "gpu": "FAKE", "smoke": False},
                  open(os.path.join(d, "metrics.json"), "w"), indent=1)
        probs = np.stack([blob(*rng.integers(80, 272, 2), int(np.sqrt(a / np.pi) * SIZE)) for a in area])
        np.savez_compressed(os.path.join(d, "test_probs.npz"), names=np.array(names), probs=probs)
    print("fake results written to", OUT, "for", list(MODELS))


if __name__ == "__main__":
    main()
