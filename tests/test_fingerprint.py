"""Tests for mlmolprop.fingerprint."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mlmolprop import neutralize
from mlmolprop.fingerprint import Fingerprint, _build_fp_list, make_fingerprints


@pytest.mark.parametrize("type_f", ["ECFP4", "maccs", "MACCs", "all"])
def test_make_fingerprints_valid_types(small_molecules, type_f):
    names = [[m.GetProp("_Name") for m in small_molecules]]
    df, bit_info = make_fingerprints(small_molecules, names, length=64, type_f=type_f)
    assert df.shape[0] == len(small_molecules)
    assert isinstance(bit_info, dict)


def test_make_fingerprints_all_preserves_every_fingerprint_type(small_molecules):
    names = [[m.GetProp("_Name") for m in small_molecules]]
    df, _ = make_fingerprints(small_molecules, names, length=64, type_f="all")

    fp_list = _build_fp_list("all", length=64, radius=2, bit_info={})
    for fp in fp_list:
        fp.apply_fp(small_molecules)
    expected_total_columns = sum(len(fp.x[0]) for fp in fp_list)

    assert df.shape[1] == expected_total_columns


def test_fingerprint_class_apply_fp_keeps_both_halves_of_a_tuple_result(
    small_molecules,
):
    # Some RDKit fingerprint functions (e.g. the real EState FingerprintMol,
    # used internally by make_fingerprints(type_f="all")) return a
    # (counts, sums) tuple rather than a single array/bitvector. Both halves
    # are independently meaningful, so Fingerprint.apply_fp must keep both.
    from rdkit.Chem.EState.Fingerprinter import FingerprintMol

    raw = FingerprintMol(small_molecules[0])
    assert isinstance(raw, tuple) and len(raw) == 2  # confirms the precondition

    fp = Fingerprint(FingerprintMol, "estate")
    fp.apply_fp(small_molecules)

    expected_full_length = len(raw[0]) + len(raw[1])
    assert fp.x[0].shape == (expected_full_length,)


def test_make_fingerprints_identical_molecules_give_identical_rows(small_molecules):
    # Known-answer case: fingerprinting is deterministic given the same
    # molecule/parameters, so two identical molecules must produce
    # byte-identical rows. Uses the default ECFP4 mode, which the type_f
    # bug above doesn't affect (it only returns fp_list[0], and "ecfp4"
    # mode's fp_list has exactly one entry).
    duplicated = [small_molecules[0], small_molecules[0]]
    names = [["mol_a", "mol_b"]]
    df, _ = make_fingerprints(duplicated, names, length=64, type_f="ECFP4")

    assert np.array_equal(df.iloc[0].to_numpy(), df.iloc[1].to_numpy())


def test_make_fingerprints_invalid_type_raises(small_molecules):
    names = [[m.GetProp("_Name") for m in small_molecules]]
    with pytest.raises(ValueError):
        make_fingerprints(small_molecules, names, type_f="bogus")


def test_make_fingerprints_bit_info_not_shared_across_calls(small_molecules):
    # regression guard: bit_info used to be a module-level global dict,
    # silently shared and never reset across separate calls
    names = [[m.GetProp("_Name") for m in small_molecules]]
    _, bit_info_1 = make_fingerprints(small_molecules, names, length=64, type_f="ECFP4")
    _, bit_info_2 = make_fingerprints(
        small_molecules[:1], names, length=64, type_f="ECFP4"
    )
    assert bit_info_1 is not bit_info_2


def test_fingerprint_class_apply_fp(small_molecules):
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

    fp = Fingerprint(lambda m: GetMorganFingerprintAsBitVect(m, 2, nBits=32), "morgan")
    fp.apply_fp(small_molecules)
    assert len(fp.x) == len(small_molecules)
    assert all(arr.shape == (32,) for arr in fp.x)


@pytest.mark.parametrize(
    "drawn, neutral",
    [
        ("CC(=O)[O-]", "CC(=O)O"),  # carboxylate -> acid
        ("CC[NH3+]", "CCN"),  # ammonium -> amine
        ("[O-]c1ccccc1", "Oc1ccccc1"),  # phenolate -> phenol
        ("[NH3+]CC(=O)[O-]", "NCC(=O)O"),  # zwitterion -> neutral
        ("C[N+](C)(C)C", "C[N+](C)(C)C"),  # quaternary ammonium has no neutral form
        ("O=[N+]([O-])c1ccccc1", "O=[N+]([O-])c1ccccc1"),  # nitro stays charge-separated
        ("CCO", "CCO"),  # already neutral
    ],
)
def test_neutralize_gives_the_neutral_form_where_one_exists(drawn, neutral):
    out = neutralize(Chem.MolFromSmiles(drawn))
    assert Chem.MolToSmiles(out) == Chem.MolToSmiles(Chem.MolFromSmiles(neutral))


def test_neutralize_keeps_input_name_and_2d_layout():
    mol = Chem.MolFromSmiles("OC(=O)c1ccccc1C(=O)[O-]")
    mol.SetProp("_Name", "phthalate")
    AllChem.Compute2DCoords(mol)
    before = Chem.MolToSmiles(mol)
    coords = mol.GetConformer().GetPositions()

    out = neutralize(mol)

    assert Chem.MolToSmiles(mol) == before  # the input is not modified
    assert out.GetProp("_Name") == "phthalate"
    assert [a.GetSymbol() for a in out.GetAtoms()] == [a.GetSymbol() for a in mol.GetAtoms()]
    np.testing.assert_allclose(out.GetConformer().GetPositions(), coords)


def test_make_fingerprints_neutralize_matches_fingerprints_of_the_neutral_forms():
    drawn = [Chem.MolFromSmiles(s) for s in ("CC(=O)[O-]", "CC[NH3+]", "c1ccccc1")]
    neutral = [Chem.MolFromSmiles(s) for s in ("CC(=O)O", "CCN", "c1ccccc1")]
    names = [["a", "b", "c"]]
    before = [Chem.MolToSmiles(m) for m in drawn]

    as_drawn, _ = make_fingerprints(drawn, names, length=256)
    neutralized, _ = make_fingerprints(drawn, names, length=256, neutralize=True)
    expected, _ = make_fingerprints(neutral, names, length=256)

    pd.testing.assert_frame_equal(neutralized, expected)
    assert not as_drawn.equals(neutralized)  # the default fingerprints the molecules as drawn
    assert [Chem.MolToSmiles(m) for m in drawn] == before  # the inputs are not modified
