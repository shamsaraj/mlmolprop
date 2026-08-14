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


def _make_name_position_csv(path, name_position: int, rng):
    # 4 columns: featA, name, activity, featB, reordered so NAME sits at
    # `name_position`. featA/featB get distinctive, easily-checked values.
    n = 40
    cols = {
        "featA": np.arange(1000, 1000 + n, dtype=float),
        "name": [f"m{i}" for i in range(n)],
        "activity": rng.integers(0, 2, size=n),
        "featB": np.arange(-n, 0, dtype=float),
    }
    order = [c for c in cols if c != "name"]
    order.insert(name_position, "name")
    pd.DataFrame({c: cols[c] for c in order}).to_csv(path, index=False)


def test_data_prep_imputation_v_names_correct_when_name_is_first_column(
    cwd_tmp_path, rng
):
    # NAME as the first column is the one arrangement that already works
    # correctly today (a passing baseline, not part of the bug).
    csv_path = "data.csv"
    _make_name_position_csv(csv_path, name_position=0, rng=rng)

    result = data_prep(
        csv_path, imputation="on", NAME="name", TARGET="activity",
        mod="class", ratio=0.25, rs=0,
    )
    v_names = result[4]
    assert set(v_names) == {"featA", "featB"}


@pytest.mark.parametrize("name_position", [1, 2, 3])
def test_data_prep_imputation_v_names_correct_regardless_of_name_column_position(
    cwd_tmp_path, rng, name_position
):
    # Property: v_names (the real feature columns) should be the same set
    # -- {featA, featB} -- no matter where NAME happens to sit in the CSV.
    # (name_position=0 is covered separately above -- it's the one case
    # that isn't buggy.)
    csv_path = "data.csv"
    _make_name_position_csv(csv_path, name_position, rng)

    result = data_prep(
        csv_path, imputation="on", NAME="name", TARGET="activity",
        mod="class", ratio=0.25, rs=0,
    )
    v_names = result[4]
    assert set(v_names) == {"featA", "featB"}


def test_data_prep_imputation_name_last_column_extreme_case(cwd_tmp_path, rng):
    # Edge case: NAME as the very LAST column -- every other column sits
    # before it. Regression guard for the (now-fixed) column-mislabeling
    # bug: v_names must still be {featA, featB}, and TARGET must keep its
    # real 0/1 activity values rather than being overwritten by featA's
    # (1000+) marker values -- which used to make the stratified split
    # raise "least populated class has only 1 member" instead of
    # completing normally.
    csv_path = "data.csv"
    _make_name_position_csv(csv_path, name_position=3, rng=rng)

    result = data_prep(
        csv_path, imputation="on", NAME="name", TARGET="activity",
        mod="class", ratio=0.25, rs=0,
    )
    v_names = result[4]
    y_train, y_test = result[1], result[3]
    assert set(v_names) == {"featA", "featB"}
    assert set(np.concatenate([y_train, y_test])) <= {0, 1}


def test_data_prep_imputation_name_not_first_known_answer(cwd_tmp_path, rng):
    # Known-answer case, hand-traced and confirmed by direct execution:
    # columns [featA, name, activity, featB] (NAME at position 1) ->
    # reconstruction drops position 0 ("featA") from the label list, so
    # v_names currently comes out as ['name', 'featB'] -- featA's data is
    # silently relabeled "name", and "activity" happens to stay correctly
    # aligned since it comes after NAME's original position (only columns
    # *before* NAME get corrupted). Documents the exact, verified failure.
    csv_path = "data.csv"
    _make_name_position_csv(csv_path, name_position=1, rng=rng)

    result = data_prep(
        csv_path, imputation="on", NAME="name", TARGET="activity",
        mod="class", ratio=0.25, rs=0,
    )
    v_names = result[4]
    assert "featA" in v_names
    assert "name" not in v_names


def test_data_prep_rowremoval_filters_rows(dataset_csv):
    result = data_prep(
        dataset_csv, rowremoval=1, TARGET="IC50", mod="class", ratio=0.25, rs=0
    )
    y_train, y_test = result[1], result[3]
    assert set(y_train) | set(y_test) == {0}


def test_data_prep_fits_transformers_on_train_rows_only(
    dataset_with_categorical_csv, monkeypatch
):
    # regression guard for the data-leakage bug: VarianceThreshold,
    # StandardScaler, correlation filtering, and SelectKBest all used to be
    # fit on the *full* pre-split dataset (SelectKBest even saw the test
    # set's own target values), biasing every reported R2_test/Q2_CV
    # optimistically. Spy on each fitted call and assert it only ever sees
    # as many rows as X_train actually has -- a precise, non-statistical
    # guard rather than relying on noisy end-to-end metric comparisons.
    from mlmolprop import processing as processing_module

    vt_calls, scaler_calls, selectkbest_calls, corr_calls = [], [], [], []

    original_vt_fit = processing_module.VarianceThreshold.fit
    monkeypatch.setattr(
        processing_module.VarianceThreshold,
        "fit",
        lambda self, X, *a, **kw: (
            vt_calls.append(len(X)),
            original_vt_fit(self, X, *a, **kw),
        )[1],
    )

    original_scaler_fit = processing_module.preprocessing.StandardScaler.fit
    monkeypatch.setattr(
        processing_module.preprocessing.StandardScaler,
        "fit",
        lambda self, X, *a, **kw: (
            scaler_calls.append(len(X)),
            original_scaler_fit(self, X, *a, **kw),
        )[1],
    )

    original_skb_fit = processing_module.SelectKBest.fit
    monkeypatch.setattr(
        processing_module.SelectKBest,
        "fit",
        lambda self, X, y, *a, **kw: (
            selectkbest_calls.append((len(X), len(y))),
            original_skb_fit(self, X, y, *a, **kw),
        )[1],
    )

    original_find_correlation = processing_module.find_correlation
    monkeypatch.setattr(
        processing_module,
        "find_correlation",
        lambda data, *a, **kw: (
            corr_calls.append(len(data)),
            original_find_correlation(data, *a, **kw),
        )[1],
    )

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
    X_train = result[0]
    n_train = len(X_train)

    assert vt_calls == [n_train]
    assert scaler_calls == [n_train]
    assert corr_calls == [n_train]
    assert selectkbest_calls == [(n_train, n_train)]


def test_data_prep_scaler_stats_match_train_rows_not_full_dataset(dataset_csv):
    # Independent check of the same leakage fix: the fitted StandardScaler's
    # per-column mean should equal the mean of just the training rows
    # (identified via the returned X_train's own index into the raw CSV),
    # not the mean over the full dataset.
    # data_prep only uses the "name" column as the row index when
    # imputation="on"; by default (as here) it's dropped and the plain
    # RangeIndex from read_csv survives through to X_train.index.
    raw = pd.read_csv(dataset_csv)
    result = data_prep(dataset_csv, Scaled="on", mod="reg", ratio=0.25, rs=0)
    X_train, y_train, X_test, y_test, v_names, Xnormalized, Xscaled, v_names2 = result

    train_rows = raw.loc[X_train.index]
    for i, col in enumerate(v_names2):
        assert Xscaled.mean_[i] == pytest.approx(train_rows[col].mean())
        assert Xscaled.mean_[i] != pytest.approx(raw[col].mean())


@pytest.fixture
def bonferroni_dataset_csv(rng, cwd_tmp_path):
    # f1_dup is near-perfectly correlated with f1, so Cor="on" drops
    # exactly one of them -- leaving 3 real features (f1_dup, f2, f3) at
    # SelectKBest time, down from 4 before correlation filtering.
    n = 60
    f1 = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "name": [f"m{i}" for i in range(n)],
            "f1": f1,
            "f1_dup": f1 * 2 + rng.normal(scale=0.0001, size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
            "activity": rng.integers(0, 2, size=n),
        }
    )
    df.to_csv("bonferroni_data.csv", index=False)
    return "bonferroni_data.csv"


def _bonferroni_multiplier(monkeypatch, csv_path, **data_prep_kwargs):
    """Run data_prep with FS="clas", returning (implied_multiplier, n_at_fit)."""
    from mlmolprop import processing as processing_module

    captured = {}
    original_fit = processing_module.SelectKBest.fit

    def spy_fit(self, X, y, *a, **kw):
        result = original_fit(self, X, y, *a, **kw)
        captured["pvalues"] = result.pvalues_.copy()
        captured["n_at_fit"] = X.shape[1]
        return result

    monkeypatch.setattr(processing_module.SelectKBest, "fit", spy_fit)

    data_prep(
        csv_path, TARGET="activity", FS="clas", mod="class", ratio=0.25, rs=0,
        **data_prep_kwargs,
    )
    bonferroni_df = pd.read_csv("Bonferroni.csv", index_col=0)
    implied_multiplier = bonferroni_df["p value"].to_numpy() / captured["pvalues"]
    return implied_multiplier, captured["n_at_fit"]


def test_data_prep_bonferroni_multiplier_matches_features_actually_tested(
    monkeypatch, bonferroni_dataset_csv
):
    # Property: the multiplier baked into Bonferroni.csv should equal the
    # number of features SelectKBest actually saw, whatever that is.
    implied_multiplier, n_at_fit = _bonferroni_multiplier(
        monkeypatch, bonferroni_dataset_csv, Cor="on"
    )
    assert implied_multiplier == pytest.approx(n_at_fit)


def test_data_prep_bonferroni_multiplier_correct_without_correlation_filtering(
    monkeypatch, bonferroni_dataset_csv
):
    # Edge case: with Cor="off" (the default), v_names2 and the actual
    # SelectKBest feature count never diverge, so the multiplier is
    # correct today -- confirms the bug is specifically about Cor="on"
    # changing the feature count out from under the stale v_names2, not
    # about the Bonferroni formula being wrong in general.
    implied_multiplier, n_at_fit = _bonferroni_multiplier(
        monkeypatch, bonferroni_dataset_csv
    )
    assert implied_multiplier == pytest.approx(n_at_fit)


def test_data_prep_bonferroni_multiplier_known_answer(monkeypatch, bonferroni_dataset_csv):
    # Known-answer case, verified by direct execution: 4 features survive
    # VarianceThreshold (f1, f1_dup, f2, f3) -> v_names2 has length 4.
    # Cor="on" then drops f1 (correlated with f1_dup), leaving 3 features
    # at SelectKBest time. The multiplier currently used is 4, not 3.
    implied_multiplier, n_at_fit = _bonferroni_multiplier(
        monkeypatch, bonferroni_dataset_csv, Cor="on"
    )
    assert n_at_fit == 3
    assert implied_multiplier == pytest.approx(3)


@pytest.fixture
def dataset_with_missing_target_csv(rng, cwd_tmp_path):
    n = 40
    df = pd.DataFrame(
        {
            "name": [f"m{i}" for i in range(n)],
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "activity": rng.normal(loc=50, scale=5, size=n),
        }
    )
    df.loc[3, "activity"] = np.nan
    df.to_csv("target_missing.csv", index=False)
    return "target_missing.csv", df["activity"].dropna().mean()


def test_data_prep_imputation_never_alters_present_target_values(
    dataset_with_missing_target_csv,
):
    # Property (necessary but not sufficient on its own -- paired with the
    # edge case below): imputation must never change a target value that
    # was actually present. This part already holds today; SimpleImputer
    # only ever fills genuine NaNs.
    csv_path, _ = dataset_with_missing_target_csv
    original = pd.read_csv(csv_path)
    present_mask = original["activity"].notna()
    present_names = original.loc[present_mask, "name"]
    present_values = original.loc[present_mask, "activity"]

    result = data_prep(csv_path, imputation="on", TARGET="activity", mod="reg", ratio=0.25, rs=0)
    X_train, y_train, X_test, y_test = result[0], result[1], result[2], result[3]
    all_names = list(X_train.index) + list(X_test.index)
    all_y = dict(zip(all_names, list(y_train) + list(y_test)))

    for name, value in zip(present_names, present_values):
        if name in all_y:
            assert all_y[name] == pytest.approx(value)


def test_data_prep_imputation_drops_rows_with_missing_target(
    dataset_with_missing_target_csv,
):
    # Edge case: a row whose TARGET is missing should not survive into
    # the final train/test split at all -- currently it does, holding a
    # fabricated mean-imputed value.
    csv_path, _ = dataset_with_missing_target_csv
    result = data_prep(csv_path, imputation="on", TARGET="activity", mod="reg", ratio=0.25, rs=0)
    X_train, X_test = result[0], result[2]
    all_names = list(X_train.index) + list(X_test.index)
    assert "m3" not in all_names  # row 3 is the one with the missing target


def test_data_prep_imputation_missing_target_known_answer(
    dataset_with_missing_target_csv,
):
    # Known-answer case, verified by direct execution: with the fake-label
    # bug present, row 3's imputed target comes out equal to the mean of
    # the other 39 targets. Correct behavior is for that row to be absent
    # entirely, so no y value should be close to that mean at all.
    csv_path, mean_of_present = dataset_with_missing_target_csv
    result = data_prep(csv_path, imputation="on", TARGET="activity", mod="reg", ratio=0.25, rs=0)
    y_train, y_test = result[1], result[3]
    all_y = np.concatenate([y_train, y_test])
    assert len(all_y) == 39
    assert not np.any(np.isclose(all_y, mean_of_present, atol=1e-6))


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
