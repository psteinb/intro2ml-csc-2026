"""Remove standalone Marimo imports from exported Jupyter notebooks."""

import sys
from pathlib import Path

import nbformat


def main(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = [
        cell
        for cell in notebook.cells
        if not (
            cell.cell_type == "code"
            and cell.source.strip() in {"import marimo", "import marimo as mo"}
        )
    ]
    nbformat.write(notebook, path)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
