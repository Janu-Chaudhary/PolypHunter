#!/usr/bin/env bash
# Build one training notebook and run it on a Kaggle T4 GPU under YOUR Kaggle account.
# Usage: tools/run_on_kaggle.sh 01|02|03 [smoke]
# Needs the Kaggle CLI (pip install kaggle) and your API token in ~/.kaggle/ (kaggle.com -> Settings -> API).
# The Kaggle website route (upload the .ipynb, Save & Run All) does the same thing; see docs/WORK_PLAN.md.
# stop at the first failing command (-e), on unset variables (-u) and on failures inside pipes (pipefail)
set -euo pipefail
cd "$(dirname "$0")/.."                                # run from the repository root, wherever it is called from
NB=$(ls src/notebooks/"$1"_*.py | head -1)            # e.g. src/notebooks/02_pretrained_unets.py
STEM=$(basename "$NB" .py)                            # file name without folder and .py, e.g. 02_pretrained_unets
# the Kaggle user name from the local Kaggle config ("username: <name>" -> <name>)
USER_NAME=$(kaggle config view | awk -F': ' '/username/{print $2}')
SMOKE_VAL=False; SUFFIX=""
if [[ "${2:-}" == smoke ]]; then SMOKE_VAL=True; SUFFIX="-smoke"; fi   # ${2:-} = second argument, or empty
SLUG="polyphunter-${STEM//_/-}$SUFFIX"                  # e.g. polyphunter-02-pretrained-unets-smoke
python3 tools/make_notebook.py                         # rebuild the notebooks so they match src/
mkdir -p "build/$SLUG"
# The smoke variant flips the SMOKE switch in the first code cell; outputs then go to results_smoke/.
sed "s/SMOKE = False/SMOKE = $SMOKE_VAL/" "notebooks/$STEM.ipynb" > "build/$SLUG/$STEM.ipynb"
# Kaggle kernel settings: private notebook, T4 GPU, internet on (pip install and pretrained weights),
# and the three datasets mounted under /kaggle/input
cat > "build/$SLUG/kernel-metadata.json" <<EOF
{"id": "$USER_NAME/$SLUG", "title": "$SLUG", "code_file": "$STEM.ipynb", "language": "python",
 "kernel_type": "notebook", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "machine_shape": "NvidiaTeslaT4",
 "dataset_sources": ["debeshjha1/kvasirseg", "debeshjha1/kvasirsessile", "balraj98/cvcclinicdb"],
 "competition_sources": [], "kernel_sources": []}
EOF
kaggle kernels push -p "build/$SLUG"                   # upload and start a new version of the notebook
echo "When it finishes: kaggle kernels output $USER_NAME/$SLUG -p build/$SLUG/output"
