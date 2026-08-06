"""Tests for mlmolprop.processing."""

import numpy as np
import pandas as pd
import pytest

from mlmolprop.processing import (
    SelectKBest_selector,
    VarianceThreshold_selector,
    average_bygroup,
    data_prep,
    file2list,
    file2object,
    list2file,
    object2file,
)


@pytest.fixture
def dataset_csv(rng, cwd_tmp_path):
    n = 60
    df = pd.DataFrame(
        {
            "name": [f"m{i}" for i in range(n)],
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
            "const": [5.0] * n,
            "f4_corr": None,
            "IC50": rng.integers(0, 2, size=n),
        }
    )
    df["f4_corr"] = df["f1"] * 2 + rng.normal(scale=0.001, size=n)
    df.to_csv("data.csv", index=False)
    return "data.csv"


@pytest.fixture
def dataset_with_categorical_csv(dataset_csv, rng):
    df = pd.read_csv(dataset_csv)
    df["cat_col"] = rng.choice(["A", "B", "C"], size=len(df))
    df.to_csv(dataset_csv, index=False)
    return dataset_csv


def test_data_prep_basic_flow_drops_const_and_correlated(dataset_with_categorical_csv):
    result = data_prep(
        dataset_with_categorical_csv,
        Scaled="on",
        Cor="on",
        FS="reg",
        cat_columns=["cat_col"],
        mod="class",
        ratio=0.25,
        rs=0,
    )
    X_train, y_train, X_test, y_test, v_names, Xnormalized, Xscaled, v_names2 = result
    assert "const" not in X_train.columns
    assert any("cat_col" in c for c in v_names2)
    assert Xscaled is not None


def test_data_prep_newset_extracts_y_when_target_present(dataset_csv, rng):
    # regression guard: y used to be read from X *after* TARGET was already
    # dropped from X, guaranteeing a KeyError whenever a new/external set
    # included known activities
    result = data_prep(dataset_csv, mod="reg", ratio=0.25, rs=0)
    v_names2 = result[7]

    new_df = pd.DataFrame(
        {
            "name": [f"n{i}" for i in range(10)],
            "f1": rng.normal(size=10),
            "f2": rng.normal(size=10),
            "f3": rng.normal(size=10),
            "const": [5.0] * 10,
            "f4_corr": rng.normal(size=10),
            "IC50": rng.normal(size=10),
        }
    )
    new_df.to_csv("data_new.csv", index=False)

    X_new, y_new, org_v_names = data_prep(
        "data_new.csv", newset="on", v_names2=v_names2, org_v_names=v_names2
    )
    assert y_new is not None
    assert len(y_new) == 10


def test_data_prep_imputation_actually_imputes(dataset_csv, rng):
    # regression guard: SimpleImputer was called with missing_values="NaN"
    # (the string), which never matches real float NaNs, so imputation
    # silently did nothing
    df = pd.read_csv(dataset_csv)
    df.loc[0, "f1"] = np.nan
    df.loc[5, "f2"] = np.nan
    df.to_csv("data_imp.csv", index=False)

    result = data_prep("data_imp.csv", imputation="on", mod="reg", ratio=0.25, rs=0)
    assert not result[0].isnull().values.any()


def test_data_prep_rowremoval_filters_rows(dataset_csv):
    result = data_prep(
        dataset_csv, rowremoval=1, TARGET="IC50", mod="class", ratio=0.25, rs=0
    )
    y_train, y_test = result[1], result[3]
    assert set(y_train) | set(y_test) == {0}


def test_data_prep_remove_and_selection_params(dataset_csv):
    result = data_prep(dataset_csv, REMOVE=["const"], mod="reg", ratio=0.25, rs=0)
    assert "const" not in result[4]

    result2 = data_prep(
        dataset_csv, mod="reg", ratio=0.25, rs=0, selection=["f1", "f2"]
    )
    assert set(result2[4]) <= {"f1", "f2"}


def test_variance_threshold_selector_drops_constant_column():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 5, 5, 5]})
    out = VarianceThreshold_selector(df, threshold2=0)
    assert "b" not in out.columns
    assert "a" in out.columns


def test_select_kbest_selector_invalid_kind_raises(rng):
    df = pd.DataFrame(rng.normal(size=(20, 3)), columns=["a", "b", "c"])
    target = rng.integers(0, 2, size=20)
    with pytest.raises(ValueError):
        SelectKBest_selector(df, target, 2, "bogus")


def test_list2file_file2list_roundtrip(cwd_tmp_path):
    list2file(["a", "b", "c"], "list.txt")
    assert file2list("list.txt") == ["a", "b", "c"]


def test_object2file_file2object_roundtrip(cwd_tmp_path):
    obj = {"x": 1, "y": [1, 2, 3]}
    object2file(obj, "obj.pkl")
    assert file2object("obj.pkl") == obj


def test_average_bygroup(cwd_tmp_path):
    df = pd.DataFrame({"g": [1, 1, 2], "v": [10, 20, 30]})
    avr = average_bygroup(df, "g", "avr.csv")
    assert avr.loc["v", 1] == 15.0
    assert avr.loc["v", 2] == 30.0
