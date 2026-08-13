"""Convert molecule/data images to on-disk artifacts (PNG, CSV, SVG)."""

from __future__ import annotations

import os

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import rdkit.Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg


def svg_files_to_png(files, output_dir: str = ".") -> None:
    """Convert each SVG file in ``files`` to a same-named PNG in ``output_dir``."""
    try:
        import rlPyCairo  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "svg_files_to_png() requires the optional 'image' extra "
            "(rlPyCairo, reportlab's raster backend). It has no prebuilt "
            "PyPI wheel and needs a system Cairo install to build from "
            "source -- install with: pip install 'mlmolprop[image]', or "
            "use conda-forge instead, which ships a prebuilt binary: "
            "conda install -c conda-forge rlpycairo"
        ) from e

    for file in files:
        drawing = svg2rlg(file)
        name = os.path.splitext(os.path.basename(file))[0]
        renderPM.drawToFile(drawing, os.path.join(output_dir, f"{name}.png"), fmt="PNG")


def images_to_dataframe(path: str, ext: str, output: str) -> pd.DataFrame:
    """Read every ``ext`` image under ``path`` into a flattened-pixel DataFrame.

    Each image is read and flattened to a single row of pixel values,
    indexed by filename (with ``ext`` stripped). The result is written to
    ``output`` as CSV and also returned.
    """
    filenames = [f for f in os.listdir(path) if f.endswith(ext)]
    if not filenames:
        raise ValueError(f"no files ending in {ext!r} found in {path!r}")

    images = []
    names = []
    for filename in filenames:
        images.append(mpimg.imread(os.path.join(path, filename)))
        names.append(filename[: -len(ext)])

    images_array = np.array(images).reshape((len(names), -1))
    df = pd.DataFrame(images_array, index=names)
    df.to_csv(output, index=False)
    return df


def highlight(
    smiles: str, highlight_smarts: str, size: tuple[int, int] = (400, 200)
) -> str:
    """Render ``smiles`` as SVG with atoms matching ``highlight_smarts`` highlighted.

    Atom indices are labeled on the drawing. Returns the SVG markup.
    """
    mol = rdkit.Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)

    match_atoms = list(
        mol.GetSubstructMatch(rdkit.Chem.MolFromSmarts(highlight_smarts))
    )
    colors = {atom: (1, 0.35, 0.35) for atom in match_atoms}

    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    opts = drawer.drawOptions()
    for i in range(mol.GetNumAtoms()):
        opts.atomLabels[i] = mol.GetAtomWithIdx(i).GetSymbol() + str(i)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=match_atoms,
        highlightAtomColors=colors,
        highlightBonds=match_atoms,
        highlightBondColors=colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("svg:", "")
