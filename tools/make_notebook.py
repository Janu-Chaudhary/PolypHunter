"""Convert the '# %%' percent-format script into a Jupyter notebook."""
import json, re, sys
src = open(sys.argv[1]).read()
cells = []
for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    if "[markdown]" in header:
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip("\n").splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body.strip("\n")})
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[2], "w"), indent=1)
print(len(cells), "cells ->", sys.argv[2])
