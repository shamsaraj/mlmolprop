"""Tests for mlmolprop.image."""

import os

import numpy as np
import pytest
import rdkit.Chem
from rdkit.Chem.Draw import rdMolDraw2D

from mlmolprop.image import highlight, images_to_dataframe, svg_files_to_png


def test_highlight_returns_valid_svg():
    svg = highlight("[nH]1cnc2cncnc21", "ccn")
    assert "<svg" in svg


def _spy_highlight_bonds(monkeypatch):
    """Capture the actual `highlightBonds=` list DrawMolecule receives."""
    original = rdMolDraw2D.MolDraw2DSVG.DrawMolecule
    captured = {}

    def spy(self, mol, **kw):
        captured["highlightBonds"] = list(kw.get("highlightBonds", []))
        return original(self, mol, **kw)

    monkeypatch.setattr(rdMolDraw2D.MolDraw2DSVG, "DrawMolecule", spy)
    return captured


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known bug: highlight() passes `highlightBonds=match_atoms` -- the "
        "matched *atom* indices are reused directly as *bond* indices, "
        "which is a different index space. Every index actually highlighted "
        "as a bond should connect two atoms that are both in the match; "
        "reusing atom indices as bond indices highlights unrelated bonds "
        "(or, when the match is large enough, an out-of-range bond index "
        "that doesn't exist at all)."
    ),
)
def test_highlight_bonds_only_connect_matched_atoms(monkeypatch):
    # Property: propane-in-butane match -- atoms 0,1,2 matched, real
    # connecting bonds are only bond 0 (0-1) and bond 1 (1-2). Every
    # highlighted bond index must correspond to a bond between two matched
    # atoms.
    captured = _spy_highlight_bonds(monkeypatch)
    highlight("CCCC", "CCC", size=(200, 200))

    mol = rdkit.Chem.MolFromSmiles("CCCC")
    match_atoms = set(mol.GetSubstructMatch(rdkit.Chem.MolFromSmarts("CCC")))
    for bond_idx in captured["highlightBonds"]:
        bond = mol.GetBondWithIdx(bond_idx)
        assert bond.GetBeginAtomIdx() in match_atoms and bond.GetEndAtomIdx() in match_atoms


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known bug, same root cause as test_highlight_bonds_only_connect_"
        "matched_atoms: matching the whole molecule against itself gives a "
        "match_atoms list as long as the atom count, which can exceed the "
        "valid bond-index range entirely (pentane: 5 atoms but only 4 "
        "bonds). Reusing atom indices as bond indices then hands RDKit an "
        "out-of-range bond index -- this currently crashes with an "
        "RDKit-internal ValueError instead of drawing (or raising a clear, "
        "documented error)."
    ),
)
def test_highlight_full_molecule_match_does_not_crash():
    svg = highlight("CCCCC", "CCCCC", size=(200, 200))
    assert "<svg" in svg


@pytest.mark.xfail(
    strict=True,
    reason="known bug, same as test_highlight_bonds_only_connect_matched_atoms",
)
def test_highlight_known_answer_correct_bond_for_two_atom_match(monkeypatch):
    # Known-answer case: propane matched against "CC" -- RDKit's leftmost
    # match is atoms [0, 1], and the *only* real bond connecting two matched
    # atoms is bond 0 (0-1). The buggy code additionally highlights bond 1
    # (atoms 1-2), which is not part of the match at all.
    captured = _spy_highlight_bonds(monkeypatch)
    highlight("CCC", "CC", size=(200, 200))
    assert captured["highlightBonds"] == [0]


def test_images_to_dataframe_round_trips_to_csv(cwd_tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("imgs", exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(3):
        plt.imsave(f"imgs/pic{i}.png", rng.random((4, 4)), cmap="gray")

    df = images_to_dataframe("imgs/", ".png", "out.csv")
    assert df.shape[0] == 3
    assert (cwd_tmp_path / "out.csv").exists()


def test_images_to_dataframe_no_matching_files_raises(cwd_tmp_path):
    os.makedirs("empty_dir", exist_ok=True)
    with pytest.raises(ValueError):
        images_to_dataframe("empty_dir/", ".png", "out.csv")


def test_svg_files_to_png(cwd_tmp_path):
    svg = highlight("c1ccccc1", "c1ccccc1")
    with open("mol.svg", "w") as f:
        f.write(svg)
    svg_files_to_png(["mol.svg"], output_dir=".")
    assert (cwd_tmp_path / "mol.png").exists()
