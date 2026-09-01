"""Remove and detect Marimo imports in exported Jupyter notebooks."""

import argparse
import re
from pathlib import Path

import nbformat

MARIMO_IMPORT = re.compile(r"^\s*import marimo(?:\s+as\s+[A-Za-z_]\w*)?\s*$")


def find_marimo_imports(notebook: nbformat.NotebookNode) -> list[int]:
    return [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and any(MARIMO_IMPORT.fullmatch(line) for line in cell.source.splitlines())
    ]


def clean(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = "\n".join(
                line for line in cell.source.splitlines() if not MARIMO_IMPORT.fullmatch(line)
            )
    notebook.cells = [cell for cell in notebook.cells if cell.cell_type != "code" or cell.source.strip()]
    remaining_imports = find_marimo_imports(notebook)
    if remaining_imports:
        raise RuntimeError(f"{path}: Marimo imports remain in cells {remaining_imports}")
    nbformat.write(notebook, path)


def check(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    remaining_imports = find_marimo_imports(notebook)
    if remaining_imports:
        raise RuntimeError(f"{path}: Marimo imports found in cells {remaining_imports}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if a Marimo import remains")
    parser.add_argument("paths", nargs="+", type=Path)
    arguments = parser.parse_args()
    for notebook_path in arguments.paths:
        check(notebook_path) if arguments.check else clean(notebook_path)
