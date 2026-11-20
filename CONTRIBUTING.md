# Team rules — PolypHunter

These rules keep one shared repo clean while four people work on it.

## Branches
- `main` always holds working, reviewed code. **Never push directly to `main`.**
- Create a branch per task: `<name>/<topic>`, e.g. `trivedh/attention-unet`, `manoj/failure-analysis`.
- Keep branches short-lived; open a pull request (PR) as soon as the change works.

## Pull requests
- Every change reaches `main` through a PR, using the PR template.
- **At least one other team member must approve** before merging.
- Merge with **Squash and merge** (the only option enabled); the branch is deleted automatically.
- Resolve all review comments before merging.

## Commits
- Small, focused commits with clear messages in the imperative: `Add attention-gate visualisation`.
- Authors are the four team members only. Do not add AI tools or bots as authors or co-authors.

## Code and results
- All model code lives in `src/p13_polyp_segmentation.py`; the notebook is generated from it:
  `python3 tools/make_notebook.py src/p13_polyp_segmentation.py notebooks/p13_polyp_segmentation.ipynb`.
  Edit the `.py`, not the `.ipynb`, so diffs stay readable.
- Never commit datasets, model weights (`*.pt`), API keys or `kaggle.json` / `access_token`.
- Results committed to `results/` must come from a full Kaggle run (not a smoke run), with the run
  version noted in the PR.
- Fix the random seed (42) and the split (`results/split.json`) so numbers stay comparable.

## Ownership
| Area | Owner |
|---|---|
| Data pipeline, U-Net baseline, metrics | Janu Chaudhary |
| ResNet-34 / MobileNetV2 U-Nets, speed benchmark, TTA | Challa Trivedh Kumar |
| Attention U-Net, SegFormer, attention maps, threshold tuning | Debangsu Misra |
| Failure analysis, CVC-ClinicDB generalisation, report assembly | Manoj Kharkar |

The owner reviews PRs touching their area; anyone can review anything.
