"""Shared pytest fixtures for the mlmolprop test suite."""

from __future__ import annotations

import matplotlib

matplotlib.use(
    "Agg"
)  # must happen before any pyplot import, to avoid opening GUI windows

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def small_molecules():
    """A handful of simple, valid RDKit molecules with names set."""
    smiles = {
        "ethanol": "CCO",
        "benzene": "c1ccccc1",
        "acetone": "CC(=O)C",
        "aniline": "Nc1ccccc1",
        "toluene": "Cc1ccccc1",
    }
    mols = []
    for name, smi in smiles.items():
        mol = Chem.MolFromSmiles(smi)
        mol.SetProp("_Name", name)
        mols.append(mol)
    return mols


@pytest.fixture
def feature_frame(rng):
    """A small synthetic numeric feature matrix with one highly-correlated pair."""
    n = 40
    a = rng.normal(size=n)
    return pd.DataFrame(
        {
            "f1": a,
            "f2": a * 2
            + rng.normal(scale=0.001, size=n),  # ~perfectly correlated with f1
            "f3": rng.normal(size=n),
            "f4": rng.normal(size=n),
        },
        index=[f"m{i}" for i in range(n)],
    )


@pytest.fixture
def regression_xy(feature_frame, rng):
    X = feature_frame[["f1", "f3", "f4"]]
    y = pd.Series(
        X["f1"] * 2 - X["f3"] + rng.normal(scale=0.1, size=len(X)), index=X.index
    )
    return X, y


@pytest.fixture
def classification_xy(feature_frame, rng):
    X = feature_frame[["f1", "f3", "f4"]]
    score = X["f1"] * 2 - X["f3"]
    y = pd.Series((score > np.median(score)).astype(int), index=X.index)
    return X, y


@pytest.fixture
def cwd_tmp_path(tmp_path, monkeypatch):
    """Run a test with cwd set to a fresh temp directory (for functions that write files)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
