"""Prepare molecules with RDKit: salt removal, 3D embedding, and rendering.

RDimage() additionally requires the optional 'image' extra (rlPyCairo);
see its docstring.
"""

from __future__ import annotations

import os

import rdkit.Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, rdDepictor, rdPartialCharges
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.SaltRemover import SaltRemover

RDLogger.DisableLog("rdApp.*")

_SALT_REMOVER = SaltRemover()
_VALID_FORMATS = {"sdf", "smi"}


def moltosvg(mol, molSize: tuple[int, int] = (450, 150), kekulize: bool = True) -> str:
    """Render a single RDKit molecule to SVG markup."""
    mc = rdkit.Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            rdkit.Chem.Kekulize(mc)
        except Exception:  # noqa: BLE001 -- no specific Kekulize exception in RDKit
            print("can not be kekulized")
            mc = rdkit.Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("svg:", "")


def mol_prep(mol):
    """Strip salts, embed a 3D conformer, and compute Gasteiger charges for ``mol``.

    Parameters
    ----------
    mol : rdkit.Chem.Mol

    Returns
    -------
    rdkit.Chem.Mol
        A new molecule (hydrogens removed again after embedding) with a 3D
        conformer and Gasteiger partial charges (``_GasteigerCharge`` atom
        property) computed.

    Raises
    ------
    ValueError
        If salt removal, 2D coordinate generation, charge calculation, or
        3D embedding fails for this molecule.
    """
    if mol is not None and mol.HasProp("_Name"):
        name = mol.GetProp("_Name")
    else:
        name = "<unnamed>"
    try:
        stripped = _SALT_REMOVER.StripMol(mol)
        with_hs = rdkit.Chem.AddHs(stripped)
        AllChem.Compute2DCoords(with_hs)
        rdPartialCharges.ComputeGasteigerCharges(with_hs)
        AllChem.EmbedMolecule(with_hs)
        try:
            AllChem.UFFOptimizeMolecule(with_hs)
        except Exception:  # noqa: BLE001, S110 -- UFF optimization is best-effort
            pass
        return rdkit.Chem.RemoveHs(with_hs)
    except Exception as exc:
        raise ValueError(f"failed to prepare molecule {name!r}: {exc}") from exc


def _load_molecules(moleculesfile, input_format: str, delimiter: str):
    if input_format == "sdf":
        with rdkit.Chem.SDMolSupplier(moleculesfile) as suppl:
            return [mol for mol in suppl]
    if input_format == "smi":
        suppl = rdkit.Chem.SmilesMolSupplier(
            moleculesfile,
            delimiter=delimiter,
            titleLine=True,
            smilesColumn=0,
            nameColumn=1,
        )
        # NOTE: list(suppl) silently returns [] for SmilesMolSupplier in
        # some RDKit versions (confirmed: len(suppl) and a plain `for`
        # loop both work, list(suppl) does not) -- iterate explicitly.
        return [mol for mol in suppl]
    raise ValueError(
        f"input_format must be one of {sorted(_VALID_FORMATS)}, got {input_format!r}"
    )


def mol_enumerate(
    moleculesfile,
    output1: str,
    output2: str,
    input_format: str = "sdf",
    image: bool = False,
    imagepath: str | None = None,
    delimiter: str = ",",
    print_int: int = 50,
) -> list:
    """Prepare every molecule in ``moleculesfile``, writing 3D and 2D SDFs.

    Parameters
    ----------
    moleculesfile : str
        Path to the input SDF or SMILES file.
    output1, output2 : str
        Paths the 3D-prepared and 2D-rendered SDF files are written to.
    input_format : {"sdf", "smi"}, default "sdf"
    image : bool, default False
        If True, also render an SVG/TIF/GIF image of each molecule (see
        :func:`RDimage`); requires ``imagepath``.
    imagepath : str or None
        Output directory for images, required if ``image=True``.
    delimiter : str, default ","
        Field delimiter, used only when ``input_format="smi"``.
    print_int : int, default 50
        Print progress every ``print_int`` molecules.

    Returns
    -------
    list of rdkit.Chem.Mol
        The successfully prepared molecules (3D conformers intact).
        Molecules that fail preparation are skipped with a printed
        warning, so this list may be shorter than the input.
    """
    if image and not imagepath:
        raise ValueError("imagepath is required when image=True")

    input_molecules = _load_molecules(moleculesfile, input_format, delimiter)
    for i, mol in enumerate(input_molecules):
        if mol is None:
            fallback = rdkit.Chem.MolFromSmiles("c1ccccc1")
            fallback.SetProp("_Name", "dummy")
            input_molecules[i] = fallback
            print(
                f"Molecule number {i}: could not be parsed, substituted a placeholder"
            )

    print("len(input_molecules)", len(input_molecules))

    prepared_molecules = []
    with rdkit.Chem.SDWriter(output1) as w1, rdkit.Chem.SDWriter(output2) as w2:
        for i, mol in enumerate(input_molecules):
            if i % print_int == 0:
                print(i, mol.GetProp("_Name"), "to 3d")
            try:
                prepared = mol_prep(mol)
            except ValueError as exc:
                print(f"skipping molecule {i}: {exc}")
                continue

            prepared_molecules.append(prepared)
            w1.write(prepared)

            # Compute2DCoords mutates in place and would destroy the 3D
            # conformer, so write the 2D rendering from a copy. Mol(mol)
            # (not mol.ToBinary(), which drops private properties like
            # _Name) preserves everything while staying independent.
            prepared_2d = rdkit.Chem.Mol(prepared)
            AllChem.Compute2DCoords(prepared_2d)
            w2.write(prepared_2d)

    if image:
        RDimage(prepared_molecules, imagepath)
    return prepared_molecules


def RDimage(mols, output_dir: str) -> None:
    """Render each molecule to an SVG (plus TIF/GIF) file in ``output_dir``.

    Operates on copies, so the caller's molecules (including any 3D
    conformer) are left untouched.

    Requires the optional 'image' extra (rlPyCairo, reportlab's raster
    backend). It has no prebuilt PyPI wheel and needs a system Cairo
    install to build from source -- install with:
    ``pip install 'mlmolprop[image]'``, or use conda-forge instead, which
    ships a prebuilt binary: ``conda install -c conda-forge rlpycairo``.
    """
    try:
        import rlPyCairo  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "RDimage() requires the optional 'image' extra (rlPyCairo, "
            "reportlab's raster backend). It has no prebuilt PyPI wheel "
            "and needs a system Cairo install to build from source -- "
            "install with: pip install 'mlmolprop[image]', or use "
            "conda-forge instead, which ships a prebuilt binary: "
            "conda install -c conda-forge rlpycairo"
        ) from e

    from reportlab.graphics import renderPM
    from svglib.svglib import svg2rlg

    for i, mol in enumerate(mols):
        print(i, mol.GetProp("_Name"), "to svg")
        m = rdkit.Chem.Mol(mol)  # copy, not mol.ToBinary() (drops private properties)
        m = rdkit.Chem.RemoveHs(m)
        rdkit.Chem.SanitizeMol(m)
        rdkit.Chem.Kekulize(m)
        rdDepictor.Compute2DCoords(m)

        drawer = rdMolDraw2D.MolDraw2DSVG(300, 150)
        drawer.DrawMolecule(m)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace("svg:", "")

        name = os.path.join(output_dir, m.GetProp("_Name") + ".svg")
        with open(name, "w") as f:
            f.write(svg)

        drawing = svg2rlg(name)
        renderPM.drawToFile(drawing, name[:-4] + ".tif", fmt="tif")
        renderPM.drawToFile(drawing, name[:-4] + ".gif", fmt="gif")
