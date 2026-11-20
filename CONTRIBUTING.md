# Team rules: PolypHunter

These rules keep one shared repo clean while four people work on it. The full rule set is in
`CLAUDE.md`; this page is the short version for daily work.

## Branches
- `main` always holds working, reviewed code. **Never push directly to `main`.**
- One branch per issue, named `feat/<issue#>-<topic>`, for example `feat/4-pretrained-unets`.
- Keep branches short-lived; open a pull request (PR) as soon as the change works.

## Pull requests
- Every change reaches `main` through a PR that says `Closes #N` and fills every field of the template.
- **One other team member must approve** before merging. Resolve all review comments first.
- **Merge commits only.** The leader (Janu) merges; the branch is deleted automatically.

## Commits
- One commit = one logical change. Subject in the imperative mood: `Add attention-gate figure`.
  The body, if needed, explains why.
- Never amend, rebase or force-push commits that are already pushed. If a push is rejected, `git pull` first.
- Authors are the four team members only. No AI or tool attribution lines ("Co-Authored-By: …",
  "Generated with …") in commits, PRs, reviews, issues or comments. `.claude/settings.json` turns
  this off for Claude Code on every clone.

## Code and results
- Edit the sources in `src/`, then rebuild the notebooks with `python3 tools/make_notebook.py`.
  Never edit `notebooks/*.ipynb` by hand, so diffs stay readable.
- `src/polyphunter/common.py` is shared by every notebook and is frozen. Change it only through a PR
  announced to the whole team.
- Never commit datasets, model weights (`*.pt`, `weights/`), API keys or `kaggle.json` / `access_token`.
- Results committed to `results/` must come from a full Kaggle run (`SMOKE = False`), with the Kaggle
  version noted in the PR. Smoke runs write to `results_smoke/`, which is ignored.
- The random seed (42) and the split (`results/split.json`) are fixed so numbers stay comparable.

## Ownership
| Area | Owner | Issue |
|---|---|---|
| Shared pipeline (`common.py`), U-Net baseline, introduction and dataset sections | Janu Chaudhary | #2, #3 |
| ResNet-34 / MobileNetV2 U-Nets, speed benchmark, TTA | Challa Trivedh Kumar | #4 |
| Attention U-Net, SegFormer, attention maps, threshold tuning | Debangsu Misra | #5 |
| Analysis notebook, failure analysis, CVC-ClinicDB, report assembly | Manoj Kharkar | #6 |

The owner reviews PRs touching their area; anyone can review anything.

## Stuck?
Comment on your issue with the exact error text and add the `blocker` label. Do not wait silently for
more than 15 minutes.
