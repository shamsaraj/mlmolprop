"""Compute RDKit molecular descriptors and build QSAR-ready datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import (
    AllChem,
    Descriptors,
    MACCSkeys,
    PandasTools,
    SDMolSupplier,
    SmilesMolSupplier,
)
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.ML.Descriptors import MoleculeDescriptors

from .basic import makecolumn, twodlist

_SOURCE_TYPES = {"sdf", "smi", "molecule"}
_DESCRIPTOR_SETS = {"all", "positive", "interpretable"}

positive_prefixes = [
    "fr_",  # fragment counts (~110 descriptors)
    "Num",  # counts: NumAtoms, NumRings, etc.
    "MolWt",
    "ExactMolWt",
    "HeavyAtomMolWt",
    "TPSA",
    "LabuteASA",
    "SPS",
    "qed",
    "PEOE_VSA",
    "SMR_VSA",
    "SlogP_VSA",  # surface area families
]
# Filter RDKit's full descriptor list to only those starting with the above prefixes
positive_desc_names = [
    name
    for name, _ in Descriptors.descList
    if any(name.startswith(p) for p in positive_prefixes)
]

easy_prefixes = ["fr_", "Num", "MolWt", "ExactMolWt", "HeavyAtomMolWt", "TPSA"]
easy_desc_names = [
    name
    for name, _ in Descriptors.descList
    if any(name.startswith(p) for p in easy_prefixes)
]


def desc(
    molecules, source: str = "molecule", delimiter: str = ",", selected_des: str = "all"
) -> list:
    """Compute RDKit descriptors for a set of molecules.

    Parameters
    ----------
    molecules
        Path to an SDF/SMILES file, or (if ``source="molecule"``) an
        already-loaded list of RDKit Mol objects. Each molecule must have
        a ``_Name`` property set.
    source : {"sdf", "smi", "molecule"}, default "molecule"
    delimiter : str, default ","
        Field delimiter, used only when ``source="smi"``.
    selected_des : {"all", "positive", "interpretable"}, default "all"
        "positive" restricts to descriptors that are inherently
        non-negative (counts, fragment counts, surface areas, ...);
        "interpretable" restricts to a smaller, easy-to-explain subset.

    Returns
    -------
    list
        ``[names, descriptor_names, descriptor_values]``: molecule names,
        the computed descriptor names, and a list of per-molecule
        descriptor value tuples.
    """
    if source == "sdf":
        molecules_list = SDMolSupplier(molecules)
    elif source == "smi":
        molecules_list = SmilesMolSupplier(
            molecules, delimiter=delimiter, titleLine=True, smilesColumn=0, nameColumn=1
        )
    elif source == "molecule":
        molecules_list = molecules
    else:
        raise ValueError(
            f"source must be one of {sorted(_SOURCE_TYPES)}, got {source!r}"
        )

    if selected_des == "all":
        descriptor_names = [x[0] for x in Descriptors.descList]
    elif selected_des == "positive":
        descriptor_names = positive_desc_names
    elif selected_des == "interpretable":
        descriptor_names = easy_desc_names
    else:
        raise ValueError(
            f"selected_des must be one of {sorted(_DESCRIPTOR_SETS)}, got {selected_des!r}"
        )

    names = len(molecules_list) * ["null"]
    values = len(molecules_list) * ["null"]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    for i in range(len(molecules_list)):
        values[i] = calc.CalcDescriptors(molecules_list[i])
        names[i] = molecules_list[i].GetProp("_Name")

    return [names, descriptor_names, values]


def dataframe(
    desc_result,
    input_activities: str | None,
    output: str,
    target: str = "IC50",
    sdf_file: str | None = None,
    source: str = "des",
) -> pd.DataFrame:
    """Merge computed descriptors/fingerprints with an activity table into a QSAR dataset.

    Parameters
    ----------
    desc_result
        Output of :func:`desc` (``source="des"``), or raw fingerprint
        data indexable by ``pandas.DataFrame`` (``source in
        {"finger", "image"}``).
    input_activities : str or None
        Path to a CSV of activities with a "name" column matching the
        molecule names in ``desc_result``. If ``None``, the descriptor/
        fingerprint DataFrame is written out as-is -- useful for scoring
        an external set with no known activity.
    output : str
        Path the merged (or standalone) DataFrame is written to as CSV.
    target : str, default "IC50"
        Activity column to pull from ``sdf_file`` when generating
        ``input_activities`` from an SDF.
    sdf_file : str or None
        If given, activities are extracted from this SDF's "ID" and
        ``target`` fields and written to ``input_activities`` before
        merging.
    source : {"des", "finger", "image"}, default "des"
        Shape of ``desc_result``.

    Returns
    -------
    pandas.DataFrame
    """
    if sdf_file is not None:
        sdftable = PandasTools.LoadSDF(sdf_file)
        activity_df = sdftable.loc[:, ["ID", target]].rename(columns={"ID": "name"})
        activity_df.to_csv(input_activities, index=False)

    if source in ("finger", "image"):
        df1 = pd.DataFrame(desc_result)
    elif source == "des":
        df1 = pd.DataFrame(desc_result[2], index=desc_result[0], columns=desc_result[1])
    else:
        raise ValueError(
            f"source must be one of 'des', 'finger', 'image', got {source!r}"
        )

    if input_activities is not None:
        reader = pd.read_csv(input_activities)
        reader["name"] = reader["name"].astype(str)
        merged = pd.merge(reader, df1, right_index=True, left_on="name")
    else:
        merged = df1

    merged.to_csv(output, index=False)
    return merged


def tanimoto(molecules_list, names, output: str, kind: str = "fps") -> pd.DataFrame:
    """Compute a pairwise Tanimoto similarity matrix for a list of molecules.

    Parameters
    ----------
    molecules_list : list of rdkit.Chem.Mol
    names : list
        Molecule names/labels, used as the DataFrame's index and columns.
    output : str
        Path the similarity matrix is written to as CSV.
    kind : {"fps", "maccs", "ecfp", "fcfp"}, default "fps"
        Fingerprint used for similarity: Daylight ("fps"), MACCS keys,
        Morgan/ECFP, or feature-based Morgan/FCFP.
    """
    if kind == "fps":
        fps = [FingerprintMols.FingerprintMol(m) for m in molecules_list]
    elif kind == "maccs":
        fps = [MACCSkeys.GenMACCSKeys(m) for m in molecules_list]
    elif kind == "ecfp":
        fps = [AllChem.GetMorganFingerprint(m, 6) for m in molecules_list]
    elif kind == "fcfp":
        fps = [
            AllChem.GetMorganFingerprint(m, 6, useFeatures=True) for m in molecules_list
        ]
    else:
        raise ValueError(
            f"kind must be one of 'fps', 'maccs', 'ecfp', 'fcfp', got {kind!r}"
        )

    n = len(fps)
    table = twodlist(n, n)
    for i, x in enumerate(fps):
        for j, y in enumerate(fps):
            table[i][j] = DataStructs.TanimotoSimilarity(x, y)

    df = pd.DataFrame(table, index=names, columns=names)
    df.to_csv(output)
    return df


def CI(
    X_train, X_test, v_names, train_names, test_names, cutoff: float
) -> list[pd.DataFrame]:
    """Standardize train/test features against training-set statistics and flag outliers.

    For each feature, z-scores (``|value - train_mean| / train_std``) are
    computed for both train and test sets using the *training* set's mean
    and standard deviation (features with zero training variance get a
    z-score of 0). ``out_train``/``out_test`` keep only the z-scores that
    exceed ``cutoff``, leaving the placeholder string ``"none"``
    elsewhere -- matching the original R ``CI_Test`` this was ported from.

    Returns
    -------
    list[pandas.DataFrame]
        ``[table_train, table_test, out_train, out_test]``, each a
        z-score table indexed by ``train_names``/``test_names``.
    """
    train = np.array(X_train)
    test = np.array(X_test)
    r, c = len(train), len(train[0])
    r2 = len(test)

    table_train = twodlist(r, c)
    table_test = twodlist(r2, c)
    out_train = twodlist(r, c)
    out_test = twodlist(r2, c)

    for i in range(r):
        table_train[i] = train[i]
        if i < r2:
            table_test[i] = test[i]

    column = makecolumn(table_train, c)
    col_means = [np.mean(column[j]) for j in range(c)]
    col_stds = [np.std(column[j]) for j in range(c)]

    for i in range(r):
        for j in range(c):
            if col_stds[j] != 0:
                table_train[i][j] = np.abs(
                    (table_train[i][j] - col_means[j]) / col_stds[j]
                )
                if i < r2:
                    table_test[i][j] = np.abs(
                        (table_test[i][j] - col_means[j]) / col_stds[j]
                    )
            else:
                table_train[i][j] = 0
                if i < r2:
                    table_test[i][j] = 0

    for i in range(r):
        for j in range(c):
            if table_train[i][j] > cutoff:
                out_train[i][j] = table_train[i][j]
                if i < r2 and table_test[i][j] > cutoff:
                    out_test[i][j] = table_test[i][j]

    return [
        pd.DataFrame(table_train, columns=v_names, index=train_names),
        pd.DataFrame(table_test, columns=v_names, index=test_names),
        pd.DataFrame(out_train, columns=v_names, index=train_names),
        pd.DataFrame(out_test, columns=v_names, index=test_names),
    ]
