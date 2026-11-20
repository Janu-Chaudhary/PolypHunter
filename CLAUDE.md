# PolypHunter: project rules

P13 deep-learning group project: polyp segmentation on Kvasir-SEG with U-Net and its variants.
Team: Janu Chaudhary (leader), Challa Trivedh Kumar, Debangsu Misra, Manoj Kharkar.
Deliverables: the project report and this repository. Deadline: Sun 22 Nov 2026.

## Git rules
- One commit = one logical change. Never mix unrelated edits.
- Subject line in the imperative mood, saying what the commit does: `Add attention-gate figure`.
  The body, if needed, explains **why**, not a restatement of the diff.
- Never rewrite published history (no rebase or amend on pushed commits). Never force-push.
  If a push is rejected, `git pull` first.
- Never commit `.env`, credentials, keys, `kaggle.json` / `access_token`, datasets, model weights
  (`*.pt`), build outputs or dependency folders.
- Never add author, co-author or tool attribution lines (for example "Co-Authored-By: Claude …" or
  "🤖 Generated with Claude Code") to commit messages, PR descriptions, reviews, issues or comments.
  `.claude/settings.json` sets `"attribution": false` so this holds on every teammate's machine.

## Team conventions
- One branch and one PR per issue. Branch name: `feat/<issue#>-<topic>`, for example `feat/4-pretrained-unets`.
- The PR body says `Closes #N` and fills every field of the PR template.
- Merge commits only. The leader (Janu) merges, after one teammate's approval.
- Each member edits only the files their issue lists. `src/polyphunter/common.py` is frozen: change it
  only through a PR announced to the whole team, because every notebook depends on it.

## Code comments
- Every file starts with a short header: what it does and how it fits the project.
- Every function gets a docstring: purpose, inputs, output, and any invariant it relies on.
- Comment the **why** and the method step from the paper (for example
  "attention gate: alpha = sigmoid(psi(relu(Wg*g + Wx*x)))"), not the syntax. State what each key variable holds.
- Explain every non-obvious index, size or boundary (resolution after pooling, channel counts, thresholds) in one line.
- Plain English that can be read aloud in a viva. No commented-out code. Update comments in the same edit as the code.

## How to run
- After editing anything in `src/`, rebuild the notebooks: `python3 tools/make_notebook.py`
  (writes `notebooks/0X_*.ipynb`; never edit the `.ipynb` files by hand).
- Train on Kaggle: upload `notebooks/0X_*.ipynb`, add the datasets `debeshjha1/kvasirseg`,
  `debeshjha1/kvasirsessile`, `balraj98/cvcclinicdb`, set GPU T4 and Internet On, then Save & Run All.
  From the command line instead: `tools/run_on_kaggle.sh 01|02|03 [smoke]`.
- Analysis without a GPU: `python3 tools/make_fake_results.py`, then
  `RESULTS_DIR=results/_fake python3 src/notebooks/04_analysis.py` (or run `notebooks/04_analysis.ipynb`).
- Check before a PR: `python3 -m py_compile src/polyphunter/common.py src/notebooks/*.py tools/*.py`.

## Directory map
- `src/polyphunter/common.py`: shared data loading, split, augmentation, models, training loop, metrics,
  FPS benchmark and result writer.
- `src/notebooks/01_unet_baseline.py` (Janu), `02_pretrained_unets.py` (Trivedh),
  `03_attention_transformer.py` (Debangsu), `04_analysis.py` (Manoj): notebook sources.
- `notebooks/`: notebooks generated from `src/`; `common.py` is inlined so Kaggle needs no extra files.
- `results/<model>/`: the output format every training notebook writes (see `docs/WORK_PLAN.md`).
- `figures/`: figures used in the report.
- `logs/`: the Kaggle log of every run behind a committed result, as text (`tools/kaggle_log_to_text.py`).
- `report/sections/`: one Markdown file per report section; `report/REPORT.md` is the assembled report.
- `docs/WORK_PLAN.md`: roles, flow, timeline and step-by-step instructions.
- `tools/`: notebook builder, Kaggle launcher, fake-results generator, Kaggle log converter.
