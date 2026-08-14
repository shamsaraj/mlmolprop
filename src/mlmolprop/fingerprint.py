"""Compute molecular fingerprints (ECFP/Morgan, MACCS, Avalon, etc.) via RDKit."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import rdkit.DataStructs.cDataStructs
from rdkit import RDLogger
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetErGFingerprint, GetMorganFingerprintAsBitVect
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdMolDescriptors import (
    GetHashedAtomPairFingerprintAsBitVect,
    GetHashedTopologicalTorsionFingerprintAsBitVect,
    GetMACCSKeysFingerprint,
)
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

_VALID_TYPES = {"ecfp4", "maccs", "all"}


def explicit_bitvect_to_numpy_array(
    bitvector: rdkit.DataStructs.cDataStructs.ExplicitBitVect,
) -> np.ndarray:
    """Convert an RDKit ExplicitBitVect to a numpy array of 0/1 values."""
    arr = np.zeros((0,), dtype=np.int8)
    ConvertToNumpyArray(bitvector, arr)
    return arr


class Fingerprint:
    """A named fingerprint function plus the results accumulated by applying it."""

    def __init__(self, fp_fun, name: str):
        self.fp_fun = fp_fun
        self.name = name
        self.x: list[np.ndarray] = []

    def apply_fp(self, mols) -> None:
        for mol in mols:
            fp = self.fp_fun(mol)
            if isinstance(fp, tuple):
                # e.g. the real EState FingerprintMol, which returns
                # (counts, sums) -- both halves are independently
                # meaningful, so keep both rather than just fp[0].
                fp = np.concatenate([np.asarray(part) for part in fp])
            elif isinstance(fp, rdkit.DataStructs.cDataStructs.ExplicitBitVect):
                fp = explicit_bitvect_to_numpy_array(fp)
            elif isinstance(fp, rdkit.DataStructs.cDataStructs.IntSparseIntVect):
                fp = np.array(list(fp))
            self.x.append(fp)


def _build_fp_list(
    type_f: str, length: int, radius: int, bit_info: dict
) -> list[Fingerprint]:
    kind = type_f.strip().lower()
    if kind == "all":
        return [
            Fingerprint(
                lambda x: GetHashedAtomPairFingerprintAsBitVect(x, nBits=length),
                "Atom pair (1985)",
            ),
            Fingerprint(
                lambda x: GetHashedTopologicalTorsionFingerprintAsBitVect(
                    x, nBits=length
                ),
                "Topological torsion (1987)",
            ),
            Fingerprint(
                lambda x: GetMorganFingerprintAsBitVect(
                    x, 3, nBits=length, useFeatures=True
                ),
                "Morgan circular FCFP",
            ),
            Fingerprint(
                lambda x: GetMorganFingerprintAsBitVect(
                    x, radius=2, nBits=length, useFeatures=False, bitInfo=bit_info
                ),
                "Morgan circular ECFP",
            ),
            Fingerprint(FingerprintMol, "Estate (1995)"),
            Fingerprint(
                lambda x: GetAvalonFP(x, nBits=length), "Avalon bit based (2006)"
            ),
            Fingerprint(
                lambda x: np.append(GetAvalonFP(x, nBits=length), Descriptors.MolWt(x)),
                "Avalon+mol. weight",
            ),
            Fingerprint(GetErGFingerprint, "ErG fingerprint (2006)"),
            Fingerprint(
                lambda x: RDKFingerprint(x, fpSize=length), "RDKit fingerprint"
            ),
            Fingerprint(GetMACCSKeysFingerprint, "MACCS"),
            Fingerprint(FingerprintMols.FingerprintMol, "Daylight fingerprint"),
        ]
    if kind == "ecfp4":
        return [
            Fingerprint(
                lambda x: GetMorganFingerprintAsBitVect(
                    x, radius=radius, nBits=length, useFeatures=False, bitInfo=bit_info
                ),
                "Morgan circular ECFP",
            )
        ]
    if kind == "maccs":
        return [Fingerprint(GetMACCSKeysFingerprint, "MACCS")]
    raise ValueError(f"type_f must be one of {sorted(_VALID_TYPES)}, got {type_f!r}")


def make_fingerprints(
    data,
    data_list,
    length: int = 256,
    verbose: bool = False,
    type_f: str = "ECFP4",
    radius: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """Compute one or more fingerprint types for a list of RDKit molecules.

    Parameters
    ----------
    data : list of rdkit.Chem.Mol
        Molecules to fingerprint.
    data_list : list
        Parallel metadata; ``data_list[0]`` is used as the row index of the
        returned DataFrame.
    length : int, default 256
        Fingerprint bit-vector length (``nBits``), where applicable.
    verbose : bool, default False
        Print progress as each fingerprint type is computed.
    type_f : str, default "ECFP4"
        Which fingerprint(s) to compute: ``"ECFP4"``, ``"MACCS"``, or
        ``"all"`` (case-insensitive).
    radius : int, default 2
        Morgan fingerprint radius, used only when ``type_f="ECFP4"``.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        The fingerprint DataFrame (indexed by ``data_list[0]``, columns
        prefixed by fingerprint name -- e.g. ``"MACCS_0"`` -- when
        ``type_f="all"`` computes more than one kind) and the Morgan
        bit-info dict populated for the last molecule processed.
    """
    bit_info: dict = {}
    fp_list = _build_fp_list(type_f, length, radius, bit_info)

    for fp in fp_list:
        if verbose:
            print("doing", fp.name)
        fp.apply_fp(data)

    if len(fp_list) == 1:
        df = pd.DataFrame(data=fp_list[0].x, index=data_list[0])
    else:
        columns = [
            f"{fp.name}_{i}" for fp in fp_list for i in range(len(fp.x[0]))
        ]
        rows = [
            np.concatenate([fp.x[row] for fp in fp_list]) for row in range(len(data))
        ]
        df = pd.DataFrame(data=rows, index=data_list[0], columns=columns)
    return df, bit_info
