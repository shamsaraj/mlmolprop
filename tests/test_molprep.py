"""Tests for mlmolprop.molprep."""

import pytest
from rdkit import Chem

from mlmolprop.molprep import RDimage, mol_enumerate, mol_prep, moltosvg


def test_moltosvg_produces_valid_svg():
    mol = Chem.MolFromSmiles("CCO")
    svg = moltosvg(mol)
    assert "<svg" in svg


def test_mol_prep_produces_3d_conformer():
    mol = Chem.MolFromSmiles("c1ccccc1O")
    mol.SetProp("_Name", "phenol")
    prepared = mol_prep(mol)
    assert prepared.GetConformer().Is3D()


def test_mol_prep_raises_instead_of_substituting_fake_molecule():
    # regression guard: mol_prep used to silently return a hardcoded benzene
    # ring named "dummy" on any failure, with a broken except block that
    # referenced an undefined variable
    class FakeBadMol:
        def HasProp(self, name):
            return name == "_Name"

        def GetProp(self, name):
            return "bad_mol"

    with pytest.raises(ValueError, match="bad_mol"):
        mol_prep(FakeBadMol())


def test_mol_enumerate_from_smi_file(cwd_tmp_path):
    # regression guard: list(SmilesMolSupplier(...)) silently returned an
    # empty list in some RDKit versions; mol_enumerate now iterates
    # explicitly instead
    import pandas as pd

    smi_df = pd.DataFrame(
        {
            "smiles": ["CCO", "c1ccccc1", "CCN"],
            "name": ["ethanol", "benzene", "ethylamine"],
        }
    )
    smi_df.to_csv("mols.smi", index=False)

    prepared = mol_enumerate("mols.smi", "out3d.sdf", "out2d.sdf", input_format="smi")
    assert len(prepared) == 3


def test_mol_enumerate_keeps_3d_conformer_in_returned_molecules(cwd_tmp_path):
    # regression guard: Compute2DCoords() for the 2D output file used to
    # mutate the same molecule objects in place, destroying the 3D
    # conformer on what mol_enumerate returns to the caller
    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1", "CC(N)=O"]]
    for i, m in enumerate(mols):
        m.SetProp("_Name", f"mol{i}")
    with Chem.SDWriter("in.sdf") as w:
        for m in mols:
            w.write(m)

    prepared = mol_enumerate("in.sdf", "out3d.sdf", "out2d.sdf", input_format="sdf")
    assert len(prepared) == 3
    assert all(m.GetConformer().Is3D() for m in prepared)

    with Chem.SDMolSupplier("out3d.sdf") as s:
        assert all(m.GetConformer().Is3D() for m in s)
    with Chem.SDMolSupplier("out2d.sdf") as s:
        assert all(not m.GetConformer().Is3D() for m in s)


def test_mol_enumerate_skips_unpreparable_molecules_with_warning(cwd_tmp_path, capsys):
    # acetic acid is in RDKit's default SaltRemover list, so it gets
    # stripped to zero atoms -- mol_prep should raise, and mol_enumerate
    # should skip it (not substitute a fake molecule) and print a warning
    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "CC(=O)O"]]
    for i, m in enumerate(mols):
        m.SetProp("_Name", f"mol{i}")
    with Chem.SDWriter("in.sdf") as w:
        for m in mols:
            w.write(m)

    prepared = mol_enumerate("in.sdf", "out3d.sdf", "out2d.sdf", input_format="sdf")
    assert len(prepared) == 1
    assert "skipping" in capsys.readouterr().out


def test_mol_enumerate_requires_imagepath_when_image_true(cwd_tmp_path):
    import pandas as pd

    pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]}).to_csv(
        "mols.smi", index=False
    )
    with pytest.raises(ValueError):
        mol_enumerate("mols.smi", "a.sdf", "b.sdf", input_format="smi", image=True)


def test_mol_enumerate_invalid_format_raises(cwd_tmp_path):
    import pandas as pd

    pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]}).to_csv(
        "mols.smi", index=False
    )
    with pytest.raises(ValueError):
        mol_enumerate("mols.smi", "a.sdf", "b.sdf", input_format="bogus")


def test_rdimage_does_not_mutate_callers_molecules(cwd_tmp_path):
    pytest.importorskip("rlPyCairo")
    mol = Chem.MolFromSmiles("CCO")
    mol.SetProp("_Name", "ethanol")
    prepared = mol_prep(mol)
    assert prepared.GetConformer().Is3D()

    RDimage([prepared], str(cwd_tmp_path))

    assert prepared.GetConformer().Is3D()
    assert (cwd_tmp_path / "ethanol.svg").exists()
