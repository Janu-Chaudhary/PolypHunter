# PolypHunter work plan

The same content is in GitHub issue #1. Your own step-by-step instructions are in your issue:
#3 Janu, #4 Trivedh, #5 Debangsu, #6 Manoj.

## What we are building
PolypHunter segments polyps in colonoscopy images: for every pixel, the model says "polyp" or
"not polyp". This supports early colorectal cancer screening.

Dataset: Kvasir-SEG, 1000 images with hand-drawn masks. The split is fixed with seed 42:
800 train, 80 validation, 120 test.

| Step | Model (key) | Owner |
|---|---|---|
| 1 | U-Net from scratch, the baseline (`unet_scratch`) | Janu |
| 2 | U-Net + ImageNet-pretrained ResNet-34 (`unet_resnet34`) and MobileNetV2 (`unet_mobilenetv2`) | Trivedh |
| 3 | Attention U-Net (`attention_unet`) and SegFormer MiT-B2 (`segformer_mitb2`) | Debangsu |
| analysis | Combined results, accuracy vs speed, failure analysis, CVC-ClinicDB, report assembly | Manoj |

## How the work fits together: nobody waits
```
#2 Step 0 (done when merged) ──► main has common.py + 4 notebooks
        │
        ├─► #3 Janu     : run 01_unet_baseline        ─┐
        ├─► #4 Trivedh  : run 02_pretrained_unets     ─┼─► results/<model>/ (PRs) ─► #6 Manoj reruns 04_analysis
        ├─► #5 Debangsu : run 03_attention_transformer ┘                                   on real results (10 min)
        └─► #6 Manoj    : build 04_analysis on FAKE results now ───────────────────────────┘
Everyone, from minute 0: accept the invite, set up Kaggle, write your report section with placeholders.
```
- Each person trains on **their own Kaggle account**, so all runs happen at the same time.
- Each person edits **only their own files**, so there are no merge conflicts.
- Every training notebook writes the **same output format**, so the analysis is built before any real result exists.

## Output format (what every training notebook writes)
`results/split.json` and `results/test_attributes.csv` (name, area_frac, contrast, sessile) are identical
from every notebook. For each model, `results/<model>/` contains:

| File | Contents |
|---|---|
| `metrics.json` | dice, iou, precision, recall, thr_val, dice_tuned, iou_tuned, val_dice_by_threshold, dice_tta, iou_tta, cvc_dice, cvc_iou, params_M, fps_fp32, fps_fp16, fps_fp16_tta, train_min, epochs, lr, gpu, smoke |
| `history.csv` | epoch, train_loss, val_dice, val_iou |
| `per_image_test.csv` | name, dice, iou, precision, recall (one row per test image) |
| `test_probs.npz` | `names` and `probs` (uint8 0-255, shape 120×352×352) |

Weights go to `weights/` and are **never committed**.

## Timeline (IST)
| When | What |
|---|---|
| Sat 21 Nov, now | Accept the invite; Kaggle phone verification; read your papers; start your report section |
| Sat 21 Nov, after #2 merges | Training issues start |
| Sat 21 Nov, by 23:00 | Training runs done; results PRs open |
| Sun 22 Nov, 11:00 | All results merged; Manoj reruns the analysis |
| Sun 22 Nov, 15:00 | All report sections final |
| Sun 22 Nov, 19:00 | Assembled report reviewed by all 4 |
| Sun 22 Nov, 23:59 | Submit |

## Getting started (once, everyone)
1. Accept the GitHub invite: https://github.com/Janu-Chaudhary/PolypHunter/invitations or the invite email.
2. Kaggle: kaggle.com → Settings → **Phone verification**. Without it there is no GPU and no internet,
   and the notebooks cannot install packages or download pretrained weights.
3. Clone and branch:
   ```bash
   git clone https://github.com/Janu-Chaudhary/PolypHunter.git && cd PolypHunter
   git switch -c feat/<issue#>-<topic>
   ```
4. Read `CLAUDE.md` (the rules) and `CONTRIBUTING.md` (the short version).

## How to run a training notebook on Kaggle (#3, #4, #5)
1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → upload `notebooks/0X_<name>.ipynb`.
2. Right-hand panel → **Add Input** → add `debeshjha1/kvasirseg`, `debeshjha1/kvasirsessile`, `balraj98/cvcclinicdb`.
3. **Settings → Accelerator: GPU T4** and **Internet: On**.
4. Optional check: set `SMOKE = True` in the first code cell and click **Run All**. It finishes in about 5
   minutes and writes to `results_smoke/`. Then set it back to `SMOKE = False`.
5. **Save Version → Save & Run All (Commit) → Save.** It runs in the cloud, so you can close your laptop.
   Progress is under **Versions**.
6. When the status is **Complete**: open that version → **Output** → **Download All**. Copy `results/` and
   `figures/` into your clone. Do **not** copy `weights/`.
7. Commit and open the PR:
   ```bash
   git add results/<your models> results/split.json results/test_attributes.csv figures/<your figures> report/sections/<your files>
   git commit -m "Add <model> results from Kaggle run v<N>"
   git push -u origin feat/<issue#>-<topic>
   ```
   Open the PR on GitHub with `Closes #<issue#>`, fill in the template (including the Kaggle version), and
   ask one teammate to review. Janu merges.

From the command line instead of the website: `pip install kaggle`, put your token in `~/.kaggle/`, then run
`tools/run_on_kaggle.sh 01|02|03 [smoke]`.

## Rules in one breath
One branch and one PR per issue (`feat/<issue#>-<topic>`, `Closes #N`); one approval, then Janu merges with a
merge commit; imperative commit messages; never force-push; never commit data, weights or tokens; no AI
co-author lines; edit `src/`, not the `.ipynb`; `common.py` is frozen.

**Stuck for more than 15 minutes?** Comment on your issue with the exact error and add the `blocker` label.
If a training run fails, Janu's backup run has all 5 models.
