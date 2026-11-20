#!/usr/bin/env bash
# Build the notebook from src/ and run it on a Kaggle T4 GPU.  Usage: tools/run_on_kaggle.sh [smoke]
# Needs the Kaggle CLI (pip install kaggle) and an API token in ~/.kaggle/ (Kaggle -> Settings -> API).
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=src/p13_polyp_segmentation.py
if [[ "${1:-}" == smoke ]]; then
  mkdir -p build/smoke; sed 's/^SMOKE = False/SMOKE = True/' $SRC > build/smoke/src.py
  python3 tools/make_notebook.py build/smoke/src.py build/smoke/p13-smoke.ipynb
  sed -e 's#"januchaudhary/p13-polyp-segmentation"#"januchaudhary/p13-polyp-segmentation-smoke-test"#' \
      -e 's#"../notebooks/p13_polyp_segmentation.ipynb"#"p13-smoke.ipynb"#' kaggle/kernel-metadata.json > build/smoke/kernel-metadata.json
  kaggle kernels push -p build/smoke
else
  python3 tools/make_notebook.py $SRC notebooks/p13_polyp_segmentation.ipynb
  mkdir -p build/full && cp notebooks/p13_polyp_segmentation.ipynb build/full/
  sed 's#"../notebooks/p13_polyp_segmentation.ipynb"#"p13_polyp_segmentation.ipynb"#' kaggle/kernel-metadata.json > build/full/kernel-metadata.json
  kaggle kernels push -p build/full
fi
