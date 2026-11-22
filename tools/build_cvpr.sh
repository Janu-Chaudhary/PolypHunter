#!/usr/bin/env bash
# Build the CVPR-template report and an Overleaf-ready zip.
#
# Copies report/cvpr/ and the figures into build/cvpr/ (a self-contained LaTeX project), compiles it with
# Tectonic (https://tectonic-typesetting.github.io, runs BibTeX itself), writes report/REPORT_CVPR.pdf and
# zips the project to build/PolypHunter_CVPR_overleaf.zip. On Overleaf: New Project -> Upload Project ->
# choose the zip; the default pdfLaTeX compiler works.
# Usage: tools/build_cvpr.sh
# stop at the first failing command (-e), on unset variables (-u) and on failures inside pipes (pipefail)
set -euo pipefail
cd "$(dirname "$0")/.."                     # run from the repository root, wherever it is called from

OUT=build/cvpr
rm -rf "$OUT" && mkdir -p "$OUT/figures"    # start from an empty build folder every time
cp -r report/cvpr/. "$OUT/"                 # the LaTeX sources: main.tex, sec/, refs.bib and the CVPR style files
cp figures/*.png "$OUT/figures/"            # main.tex also looks in figures/ next to itself

# compile in a subshell inside build/cvpr; Tectonic reruns LaTeX and BibTeX until references settle
(cd "$OUT" && tectonic --keep-logs main.tex)
cp "$OUT/main.pdf" report/REPORT_CVPR.pdf

# Zip only the LaTeX sources and figures, not the compiled output, so Overleaf builds it fresh.
python3 - "$OUT" build/PolypHunter_CVPR_overleaf.zip <<'EOF'
import os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]                  # folder to zip, zip file to write
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(src):              # every file in the folder and its subfolders
        for f in files:
            if f.endswith((".pdf", ".log", ".aux", ".bbl", ".blg", ".out")):   # build outputs, not sources
                continue
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, src))      # store paths relative to the folder: main.tex at the zip root
print("zip:", dst)
EOF
echo "pdf: report/REPORT_CVPR.pdf"
