"""
Build the Jupyter notebooks in notebooks/ from the percent-format sources in src/notebooks/.

Each source is split into cells at lines starting with "# %%". A line "# %% COMMON" is replaced by the
cells of src/polyphunter/common.py, so every built notebook carries the shared code and Kaggle needs no
extra files. Run it after every edit in src/:  python3 tools/make_notebook.py
"""
import json, re, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent      # repository root: tools/ is one level below it
COMMON = ROOT / "src" / "polyphunter" / "common.py"


def to_cells(src):
    """Split percent-format text into notebook cells; '# %% [markdown]' cells lose their leading '# '.

    src: the whole text of one source file. Returns a list of cell dicts in the .ipynb (nbformat 4) layout."""
    cells = []
    # re.M makes ^ match at every line start, so the text is cut at each "# %%" marker line
    for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:          # [1:] drops text before the first marker
        # header = rest of the marker line (e.g. " [markdown]" or " COMMON"), body = the cell's lines
        header, _, body = chunk.partition("\n")
        if "[markdown]" in header:
            # Markdown cells are written as comments in the .py file: remove the "# " prefix of every line
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip("\n").splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
        elif header.strip() == "COMMON":
            # placeholder: insert all cells of the shared code here (recursive call on common.py)
            cells.extend(to_cells(COMMON.read_text()))
        else:
            # a code cell; execution_count and outputs stay empty because the notebook has not been run yet
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": body.strip("\n")})
    return cells


def main():
    """Build notebooks/<name>.ipynb for every src/notebooks/<name>.py."""
    out_dir = ROOT / "notebooks"; out_dir.mkdir(exist_ok=True)
    for src in sorted((ROOT / "src" / "notebooks").glob("*.py")):      # 01_ ... 04_ in order
        cells = to_cells(src.read_text())
        for i, c in enumerate(cells):
            c["id"] = f"cell-{i}"                                    # nbformat 4.5 requires a unique id per cell
        # the notebook file: the cells plus the format version and the Python 3 kernel to run them with
        nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
              "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                           "language_info": {"name": "python"}}}
        (out_dir / f"{src.stem}.ipynb").write_text(json.dumps(nb, indent=1))   # same name, .ipynb extension
        print(f"{len(cells):3d} cells -> notebooks/{src.stem}.ipynb")


if __name__ == "__main__":
    main()
