"""
Build the Jupyter notebooks in notebooks/ from the percent-format sources in src/notebooks/.

Each source is split into cells at lines starting with "# %%". A line "# %% COMMON" is replaced by the
cells of src/polyphunter/common.py, so every built notebook carries the shared code and Kaggle needs no
extra files. Run it after every edit in src/:  python3 tools/make_notebook.py
"""
import json, re, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMMON = ROOT / "src" / "polyphunter" / "common.py"


def to_cells(src):
    """Split percent-format text into notebook cells; '# %% [markdown]' cells lose their leading '# '."""
    cells = []
    for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:          # [1:] drops text before the first marker
        header, _, body = chunk.partition("\n")
        if "[markdown]" in header:
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip("\n").splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
        elif header.strip() == "COMMON":
            cells.extend(to_cells(COMMON.read_text()))
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": body.strip("\n")})
    return cells


def main():
    """Build notebooks/<name>.ipynb for every src/notebooks/<name>.py."""
    out_dir = ROOT / "notebooks"; out_dir.mkdir(exist_ok=True)
    for src in sorted((ROOT / "src" / "notebooks").glob("*.py")):
        cells = to_cells(src.read_text())
        for i, c in enumerate(cells):
            c["id"] = f"cell-{i}"                                    # nbformat 4.5 requires a unique id per cell
        nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
              "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                           "language_info": {"name": "python"}}}
        (out_dir / f"{src.stem}.ipynb").write_text(json.dumps(nb, indent=1))
        print(f"{len(cells):3d} cells -> notebooks/{src.stem}.ipynb")


if __name__ == "__main__":
    main()
