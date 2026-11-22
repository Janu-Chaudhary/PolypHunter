#!/usr/bin/env bash
# Build the CVPR-template report and an Overleaf-ready zip.
#
# Copies report/cvpr/ and the figures into build/cvpr/ (a self-contained LaTeX project), compiles it with
# Tectonic (https://tectonic-typesetting.github.io, runs BibTeX itself), writes report/REPORT_CVPR.pdf and
# zips the project to build/PolypHunter_CVPR_overleaf.zip. On Overleaf: New Project -> Upload Project ->
# choose the zip; the default pdfLaTeX compiler works.
# Usage: tools/build_cvpr.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=build/cvpr
rm -rf "$OUT" && mkdir -p "$OUT/figures"
cp -r report/cvpr/. "$OUT/"
cp figures/*.png "$OUT/figures/"            # main.tex also looks in figures/ next to itself

(cd "$OUT" && tectonic --keep-logs main.tex)
cp "$OUT/main.pdf" report/REPORT_CVPR.pdf

# Zip only the LaTeX sources and figures, not the compiled output, so Overleaf builds it fresh.
python3 - "$OUT" build/PolypHunter_CVPR_overleaf.zip <<'EOF'
import os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith((".pdf", ".log", ".aux", ".bbl", ".blg", ".out")):
                continue
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, src))
print("zip:", dst)
EOF
echo "pdf: report/REPORT_CVPR.pdf"
