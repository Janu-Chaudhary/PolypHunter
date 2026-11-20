#!/usr/bin/env bash
# Build one training notebook and run it on a Kaggle T4 GPU under YOUR Kaggle account.
# Usage: tools/run_on_kaggle.sh 01|02|03 [smoke]
# Needs the Kaggle CLI (pip install kaggle) and your API token in ~/.kaggle/ (kaggle.com -> Settings -> API).
# The Kaggle website route (upload the .ipynb, Save & Run All) does the same thing; see docs/WORK_PLAN.md.
set -euo pipefail
cd "$(dirname "$0")/.."
NB=$(ls src/notebooks/"$1"_*.py | head -1)            # e.g. src/notebooks/02_pretrained_unets.py
STEM=$(basename "$NB" .py)
USER_NAME=$(kaggle config view | awk -F': ' '/username/{print $2}')
SMOKE_VAL=False; SUFFIX=""
if [[ "${2:-}" == smoke ]]; then SMOKE_VAL=True; SUFFIX="-smoke"; fi
SLUG="polyphunter-${STEM//_/-}$SUFFIX"                  # e.g. polyphunter-02-pretrained-unets-smoke
python3 tools/make_notebook.py
mkdir -p "build/$SLUG"
# The smoke variant flips the SMOKE switch in the first code cell; outputs then go to results_smoke/.
sed "s/SMOKE = False/SMOKE = $SMOKE_VAL/" "notebooks/$STEM.ipynb" > "build/$SLUG/$STEM.ipynb"
cat > "build/$SLUG/kernel-metadata.json" <<EOF
{"id": "$USER_NAME/$SLUG", "title": "$SLUG", "code_file": "$STEM.ipynb", "language": "python",
 "kernel_type": "notebook", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "machine_shape": "NvidiaTeslaT4",
 "dataset_sources": ["debeshjha1/kvasirseg", "debeshjha1/kvasirsessile", "balraj98/cvcclinicdb"],
 "competition_sources": [], "kernel_sources": []}
EOF
kaggle kernels push -p "build/$SLUG"
echo "When it finishes: kaggle kernels output $USER_NAME/$SLUG -p build/$SLUG/output"
