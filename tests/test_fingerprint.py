"""Tests for mlmolprop.fingerprint."""

import numpy as np
import pytest

from mlmolprop.fingerprint import Fingerprint, _build_fp_list, make_fingerprints


@pytest.mark.parametrize("type_f", ["ECFP4", "maccs", "MACCs", "all"])
def test_make_fingerprints_valid_types(small_molecules, type_f):
    names = [[m.GetProp("_Name") for m in small_molecules]]
    df, bit_info = make_fingerprints(small_molecules, names, length=64, type_f=type_f)
    assert df.shape[0] == len(small_molecules)
    assert isinstance(bit_info, dict)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known bug: make_fingerprints(type_f='all') builds fp_list with "
        "all 11 fingerprint kinds and calls apply_fp() on every one of "
        "them, but the returned DataFrame is built from fp_list[0].x only "
        "(the first kind, 'Atom pair') -- the other 10 are computed then "
        "silently discarded. 'all' should be information-preserving "
        "relative to computing each kind separately; it currently returns "
        "1/11th of the data with no error."
    ),
)
def test_make_fingerprints_all_preserves_every_fingerprint_type(small_molecules):
    names = [[m.GetProp("_Name") for m in small_molecules]]
    df, _ = make_fingerprints(small_molecules, names, length=64, type_f="all")

    fp_list = _build_fp_list("all", length=64, radius=2, bit_info={})
    for fp in fp_list:
        fp.apply_fp(small_molecules)
    expected_total_columns = sum(len(fp.x[0]) for fp in fp_list)

    assert df.shape[1] == expected_total_columns


@pytest.mark.xfail(
    strict=True,
    reason=(
        "likely bug (medium confidence): some RDKit fingerprint functions "
        "(e.g. the real EState FingerprintMol, used internally by "
        "make_fingerprints(type_f='all')) return a (counts, sums) tuple "
        "rather than a single array/bitvector. Fingerprint.apply_fp "
        "special-cases tuples as `fp = np.array(list(fp[0]))`, keeping only "
        "the first element. Counts and sums are both standard, "
        "independently meaningful halves of the EState fingerprint "
        "definition (not e.g. one being auxiliary metadata), and nothing "
        "documents why only counts are kept -- reads like the same class of "
        "silent-data-drop as the type_f='all' bug, not a deliberate choice, "
        "though there's no comment either way confirming intent."
    ),
)
def test_fingerprint_class_apply_fp_keeps_both_halves_of_a_tuple_result(
    small_molecules,
):
    # Edge case: some RDKit fingerprint functions (e.g. the real EState
    # FingerprintMol, used internally by make_fingerprints(type_f="all"))
    # return a (counts, sums) tuple rather than a single array/bitvector.
    # Fingerprint.apply_fp special-cases tuples as `fp = np.array(list(fp[0]))`
    # -- keeping only the first element and silently dropping the second.
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
