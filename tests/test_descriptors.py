"""Tests for mlmolprop.descriptors."""

import pytest

from mlmolprop.descriptors import CI, dataframe, desc, tanimoto


@pytest.mark.parametrize("selected_des", ["all", "positive", "interpretable"])
def test_desc_valid_selections(small_molecules, selected_des):
    names, descriptor_names, values = desc(
        small_molecules, source="molecule", selected_des=selected_des
    )
    assert len(names) == len(small_molecules)
    assert len(descriptor_names) > 0
    assert len(values) == len(small_molecules)


def test_desc_invalid_source_raises(small_molecules):
    with pytest.raises(ValueError):
        desc(small_molecules, source="bogus")


def test_desc_invalid_selected_des_raises(small_molecules):
    with pytest.raises(ValueError):
        desc(small_molecules, selected_des="bogus")


def test_dataframe_without_activities_returns_descriptor_table(
    small_molecules, cwd_tmp_path
):
    desc_result = desc(small_molecules, source="molecule", selected_des="interpretable")
    df = dataframe(desc_result, None, "out.csv")
    assert df.shape[0] == len(small_molecules)
    assert (cwd_tmp_path / "out.csv").exists()


def test_dataframe_with_activities_merges_on_name(small_molecules, cwd_tmp_path):
    import pandas as pd

    desc_result = desc(small_molecules, source="molecule", selected_des="interpretable")
    names = desc_result[0]
    pd.DataFrame({"name": names, "IC50": range(len(names))}).to_csv(
        "act.csv", index=False
    )
    merged = dataframe(desc_result, "act.csv", "merged.csv")
    assert "IC50" in merged.columns
    assert merged.shape[0] == len(names)


def test_dataframe_invalid_source_raises(small_molecules, cwd_tmp_path):
    desc_result = desc(small_molecules, source="molecule", selected_des="interpretable")
    with pytest.raises(ValueError):
        dataframe(desc_result, None, "out.csv", source="bogus")


@pytest.mark.parametrize("kind", ["fps", "maccs", "ecfp", "fcfp"])
def test_tanimoto_all_kinds(small_molecules, kind, cwd_tmp_path):
    names = [m.GetProp("_Name") for m in small_molecules]
    df = tanimoto(small_molecules, names, "sim.csv", kind=kind)
    assert df.shape == (len(small_molecules), len(small_molecules))
    # diagonal (self-similarity) must be 1.0 for every kind, including the
    # ecfp/fcfp cases that used to crash (FingerprintSimilarity doesn't
    # support the sparse count vectors GetMorganFingerprint returns)
    for i in range(len(small_molecules)):
        assert df.iloc[i, i] == pytest.approx(1.0)


def test_tanimoto_invalid_kind_raises(small_molecules, cwd_tmp_path):
    names = [m.GetProp("_Name") for m in small_molecules]
    with pytest.raises(ValueError):
        tanimoto(small_molecules, names, "sim.csv", kind="bogus")


def test_ci_flags_outliers_using_training_stats(rng):
    n_train, n_test, n_features = 20, 5, 3
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    tables = CI(
        X_train,
        X_test,
        ["f1", "f2", "f3"],
        [f"tr{i}" for i in range(n_train)],
        [f"te{i}" for i in range(n_test)],
        cutoff=1.0,
    )
    assert len(tables) == 4
    for t in tables:
        assert t.shape[1] == n_features
