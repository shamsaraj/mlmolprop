"""Tests for chemsar.fingerprint."""

import pytest

from chemsar.fingerprint import Fingerprint, make_fingerprints


@pytest.mark.parametrize("type_f", ["ECFP4", "maccs", "MACCs", "all"])
def test_make_fingerprints_valid_types(small_molecules, type_f):
    names = [[m.GetProp("_Name") for m in small_molecules]]
    df, bit_info = make_fingerprints(small_molecules, names, length=64, type_f=type_f)
    assert df.shape[0] == len(small_molecules)
    assert isinstance(bit_info, dict)


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
