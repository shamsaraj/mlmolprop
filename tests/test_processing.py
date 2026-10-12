"""Tests for mlmolprop.processing."""

import numpy as np
import pandas as pd
import pytest

from mlmolprop.processing import (
    SelectKBest_selector,
    VarianceThreshold_selector,
    average_bygroup,
    best_without_overfitting,
    corrected_resampled_ttest,
    data_prep,
    feature_selection_cv,
    file2list,
    file2object,
    holdout_scores,
    list2file,
    object2file,
    paired_comparison,
    resplit_evaluation,
    scaffold_split,
    select_features,
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


@pytest.fixture
def features_and_activities(rng):
    # Mimics make_fingerprints()'s actual return shape: a feature matrix
    # indexed by molecule name, no NAME column of its own.
    n = 60
    names = [f"m{i}" for i in range(n)]
    features = pd.DataFrame(
        {"f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n)},
        index=names,
    )
    activities = pd.DataFrame({"name": names, "IC50": rng.integers(0, 2, size=n)})
    return features, activities


def test_data_prep_features_and_activities_matches_datafile_csv(
    features_and_activities, cwd_tmp_path
):
    features, activities = features_and_activities

    result_new = data_prep(
        features=features, activities=activities, mod="class", ratio=0.25, rs=0,
    )

    merged = features.copy()
    merged.index.name = "name"
    merged = merged.reset_index().merge(activities, on="name")
    merged.to_csv("merged.csv", index=False)
    result_csv = data_prep("merged.csv", mod="class", ratio=0.25, rs=0)

    X_train_new, y_train_new, X_test_new, y_test_new, v_names_new, *_ = result_new
    X_train_csv, y_train_csv, X_test_csv, y_test_csv, v_names_csv, *_ = result_csv

    assert v_names_new == v_names_csv
    pd.testing.assert_frame_equal(X_train_new, X_train_csv)
    pd.testing.assert_frame_equal(X_test_new, X_test_csv)
    np.testing.assert_array_equal(y_train_new, y_train_csv)
    np.testing.assert_array_equal(y_test_new, y_test_csv)


def test_data_prep_features_reused_across_two_targets(rng, cwd_tmp_path):
    n = 60
    names = [f"m{i}" for i in range(n)]
    features = pd.DataFrame(
        {"f1": rng.normal(size=n), "f2": rng.normal(size=n)}, index=names,
    )
    # Two targets over different row subsets, like our 4 CYP pIC50 columns
    # sharing one source file but different non-null molecule sets.
    activities_a = pd.DataFrame(
        {"name": names[:50], "target_a": rng.normal(size=50)}
    )
    activities_b = pd.DataFrame(
        {"name": names[10:], "target_b": rng.normal(size=50)}
    )

    result_a = data_prep(
        features=features, activities=activities_a, TARGET="target_a",
        mod="reg", ratio=0.2, rs=0,
    )
    result_b = data_prep(
        features=features, activities=activities_b, TARGET="target_b",
        mod="reg", ratio=0.2, rs=0,
    )

    assert len(result_a[0]) + len(result_a[2]) == 50
    assert len(result_b[0]) + len(result_b[2]) == 50


def test_data_prep_features_indexed_by_name_without_reset(features_and_activities, cwd_tmp_path):
    features, activities = features_and_activities
    assert "name" not in features.columns  # precondition: index-only, as documented

    result = data_prep(features=features, activities=activities, mod="class", ratio=0.25, rs=0)
    X_train = result[0]
    assert len(X_train) > 0


def test_data_prep_features_as_csv_path(features_and_activities, cwd_tmp_path):
    features, activities = features_and_activities
    features_named = features.copy()
    features_named.index.name = "name"
    features_named.reset_index().to_csv("features.csv", index=False)

    result = data_prep(
        features="features.csv", activities=activities, mod="class", ratio=0.25, rs=0,
    )
    assert len(result[0]) > 0


def test_data_prep_requires_datafile_or_features_and_activities():
    with pytest.raises(ValueError):
        data_prep()


def test_data_prep_newset_accepts_features_and_activities(features_and_activities, cwd_tmp_path):
    features, activities = features_and_activities
    result = data_prep(features=features, activities=activities, mod="class", ratio=0.25, rs=0)
    v_names2 = result[7]

    # Score a new set through the features/activities path too -- no TARGET
    # column in activities this time (unlabeled external set).
    new_names = [f"n{i}" for i in range(5)]
    new_features = pd.DataFrame(
        {"f1": [0.1] * 5, "f2": [0.2] * 5, "f3": [0.3] * 5}, index=new_names,
    )
    new_activities = pd.DataFrame({"name": new_names})

    X_new, y_new, org_v_names = data_prep(
        features=new_features, activities=new_activities,
        newset="on", v_names2=v_names2, org_v_names=v_names2,
    )
    assert y_new is None
    assert len(X_new) == 5


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


@pytest.fixture
def fs_dataset_csv(rng, cwd_tmp_path):
    """Two informative features (f1, f2) plus two pure-noise features, with both a
    continuous and a binary target derived from the same informative signal -- lets
    RFE/Elastic-Net selection tests assert the informative features actually survive
    selection over the noise ones, not just that selection runs without error.
    """
    n = 100
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    noise1 = rng.normal(size=n)
    noise2 = rng.normal(size=n)
    y_reg = 3 * f1 + 2 * f2 + rng.normal(scale=0.1, size=n)
    y_clas = (y_reg > np.median(y_reg)).astype(int)
    df = pd.DataFrame(
        {
            "name": [f"m{i}" for i in range(n)],
            "f1": f1,
            "f2": f2,
            "noise1": noise1,
            "noise2": noise2,
            "y_reg": y_reg,
            "y_clas": y_clas,
        }
    )
    df.to_csv("fs_data.csv", index=False)
    return "fs_data.csv"


def test_data_prep_fs_rfe_reg_selects_informative_features(fs_dataset_csv):
    result = data_prep(
        fs_dataset_csv, TARGET="y_reg", REMOVE=["y_clas"], FS="rfe_reg", ndes=2,
        mod="reg", ratio=0.25, rs=0,
    )
    v_names = result[4]
    assert len(v_names) == 2
    assert set(v_names) == {"f1", "f2"}


def test_data_prep_fs_rfe_clas_selects_informative_features(fs_dataset_csv):
    result = data_prep(
        fs_dataset_csv, TARGET="y_clas", REMOVE=["y_reg"], FS="rfe_clas", ndes=2,
        mod="class", ratio=0.25, rs=0,
    )
    v_names = result[4]
    assert len(v_names) == 2
    assert set(v_names) == {"f1", "f2"}


def test_data_prep_fs_enet_reg_with_fixed_k(fs_dataset_csv):
    result = data_prep(
        fs_dataset_csv, TARGET="y_reg", REMOVE=["y_clas"], FS="enet_reg", ndes=2,
        mod="reg", ratio=0.25, rs=0,
    )
    v_names = result[4]
    assert len(v_names) == 2


def test_data_prep_fs_enet_reg_without_fixed_k(fs_dataset_csv):
    # ndes=None: no top-k cap -- however many features the L1 penalty leaves
    # non-zero is however many get kept, not necessarily any particular count.
    result = data_prep(
        fs_dataset_csv, TARGET="y_reg", REMOVE=["y_clas"], FS="enet_reg", ndes=None,
        mod="reg", ratio=0.25, rs=0,
    )
    v_names = result[4]
    assert 0 < len(v_names) <= 4


def test_data_prep_fs_enet_clas_selects_features(fs_dataset_csv):
    result = data_prep(
        fs_dataset_csv, TARGET="y_clas", REMOVE=["y_reg"], FS="enet_clas", ndes=2,
        mod="class", ratio=0.25, rs=0,
    )
    v_names = result[4]
    assert len(v_names) == 2


def test_data_prep_fs_enet_clas_without_fixed_k_keeps_every_nonzero_coefficient(fs_dataset_csv):
    # ndes=None keeps every feature the penalty leaves non-zero -- not just the
    # above-average ones, SelectFromModel's default for an elastic net
    # (the reference is built the way data_prep builds it: before scikit-learn 1.8
    # LogisticRegression(l1_ratio=0.5) alone is an L2 fit, which zeroes nothing out)
    from mlmolprop.processing import _elastic_net

    kwargs = {"TARGET": "y_clas", "REMOVE": ["y_reg"], "mod": "class", "ratio": 0.25, "rs": 0}
    X_train, y_train = data_prep(fs_dataset_csv, **kwargs)[:2]
    seeded = {"random_state": 0}  # saga shuffles: the same seed gives the same coefficients
    coef = _elastic_net("clas", seeded).fit(X_train, y_train).coef_
    kept = data_prep(fs_dataset_csv, FS="enet_clas", ndes=None, enet_params=seeded, **kwargs)[4]
    assert len(kept) == int((np.abs(coef) > 1e-5).sum())


def _binary_features(rng, n, informative, noise):
    """0/1 features: ``informative`` set more often in class 1, ``noise`` unrelated to it."""
    y = rng.integers(0, 2, size=n)
    signal = rng.random((n, informative)) < np.where(y[:, None] == 1, 0.8, 0.2)
    junk = rng.random((n, noise)) < 0.3
    names = [f"i{j}" for j in range(informative)] + [f"n{j}" for j in range(noise)]
    return pd.DataFrame(np.hstack([signal, junk]).astype(int), columns=names), y


def test_feature_selection_cv_reports_every_configuration(rng):
    X, y = _binary_features(rng, n=150, informative=5, noise=25)
    configs = [("clas", 5), ("rfe_clas", 5), ("enet_clas", 5), ("enet_clas", None)]
    table = feature_selection_cv(X, y, configs, n_splits=3, rs=0)
    assert [(fs, None if pd.isna(n) else int(n)) for fs, n in zip(table["FS"], table["ndes"])] == configs
    assert (table.loc[table["FS"] != "enet_clas", "n_features"] == 5).all()
    assert table.loc[2, "n_features"] <= 5 <= table.loc[3, "n_features"] <= X.shape[1]
    assert table[["F2", "recall", "specificity", "precision", "accuracy"]].stack().between(0, 1).all()
    assert table["MCC"].between(-1, 1).all()
    assert (table["MCC"] > 0.3).all()  # the informative bits are found


def test_feature_selection_cv_is_not_optimistic_on_noise(rng):
    # Labels unrelated to the features: refitting the selection in every fold
    # keeps the cross-validated MCC near zero, while selecting once on all rows
    # lets the held-out rows choose their own features and inflates it.
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import matthews_corrcoef
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.naive_bayes import ComplementNB

    X, _ = _binary_features(rng, n=120, informative=0, noise=400)
    y = rng.integers(0, 2, size=len(X))
    in_fold = feature_selection_cv(X, y, [("clas", 10)], cor=None, n_repeats=3, rs=0).loc[0, "MCC"]
    keep = SelectKBest(f_classif, k=10).fit(X, y).get_support()
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    selected_once = matthews_corrcoef(y, cross_val_predict(ComplementNB(), X.loc[:, keep], y, cv=folds))
    assert abs(in_fold) < 0.3  # chance level (over 8 seeds: -0.12 to 0.26)
    assert selected_once > in_fold + 0.15  # (selected once: 0.32 to 0.55)


def test_feature_selection_cv_rejects_regression_selectors(rng):
    X, y = _binary_features(rng, n=40, informative=2, noise=4)
    with pytest.raises(ValueError):
        feature_selection_cv(X, y, [("reg", 2)])


def test_feature_selection_cv_return_folds_adds_per_fold_scores(rng):
    X, y = _binary_features(rng, n=150, informative=5, noise=25)
    configs = [("clas", 5), ("rfe_clas", 8)]  # deterministic selectors: the two calls must agree
    plain = feature_selection_cv(X, y, configs, n_splits=3, n_repeats=2, rs=0)
    table, folds = feature_selection_cv(X, y, configs, n_splits=3, n_repeats=2, rs=0, return_folds=True)
    pd.testing.assert_frame_equal(table, plain)
    assert len(folds) == len(configs) * 3 * 2
    assert sorted(folds["fold"].unique()) == list(range(6))
    assert (folds.groupby("fold")["repeat"].nunique() == 1).all()
    assert folds.groupby(["FS", "ndes"])["fold"].nunique().eq(6).all()
    assert (folds["n_features"] == folds["ndes"]).all()
    metrics = ["F2", "MCC", "recall", "specificity", "precision", "accuracy"]
    assert set(metrics + [f"{m}_train" for m in metrics]) <= set(folds.columns)


def test_feature_selection_cv_training_scores_show_overfitting_on_noise(rng):
    # Labels unrelated to the features: held-out MCC stays near chance, while
    # each fold's model, with features chosen on its own training rows, fits
    # those rows well.
    X, _ = _binary_features(rng, n=120, informative=0, noise=300)
    y = rng.integers(0, 2, size=len(X))
    _, folds = feature_selection_cv(X, y, [("clas", 10)], cor=None, n_repeats=2, rs=0, return_folds=True)
    assert abs(folds["MCC"].mean()) < 0.2
    assert folds["MCC_train"].mean() > folds["MCC"].mean() + 0.2


def test_corrected_resampled_ttest_identical_scores():
    a = np.array([0.4, 0.5, 0.6, 0.55, 0.45])
    assert corrected_resampled_ttest(a, a) == (0.0, 1.0)
    assert corrected_resampled_ttest(a, a, alternative="greater") == (0.0, 0.5)


def test_corrected_resampled_ttest_matches_its_formula(rng):
    from scipy import stats

    d = rng.normal(0.05, 0.1, size=10)
    t, p = corrected_resampled_ttest(d, n_splits=5)
    expected = d.mean() / np.sqrt((1 / 10 + 1 / 4) * d.var(ddof=1))
    assert t == pytest.approx(expected)
    assert p == pytest.approx(2 * stats.t(9).sf(abs(expected)))


def test_corrected_resampled_ttest_is_more_cautious_than_a_plain_paired_test(rng):
    from scipy import stats

    a = rng.normal(0.5, 0.05, size=25)
    b = a - 0.03 + rng.normal(0, 0.03, size=25)
    t, p = corrected_resampled_ttest(a, b, n_splits=5)
    assert p > stats.ttest_rel(a, b).pvalue
    greater = corrected_resampled_ttest(a, b, alternative="greater")[1]
    less = corrected_resampled_ttest(a, b, alternative="less")[1]
    assert greater + less == pytest.approx(1)
    assert min(greater, less) == pytest.approx(p / 2)
    assert corrected_resampled_ttest(a - b)[0] == pytest.approx(t)


def test_corrected_resampled_ttest_rejects_bad_input():
    with pytest.raises(ValueError):
        corrected_resampled_ttest([0.1])
    with pytest.raises(ValueError):
        corrected_resampled_ttest([0.1, 0.2], [0.0, 0.1], alternative="bigger")


def _synthetic_cv(rng, settings, n_splits=5, n_repeats=4):
    """feature_selection_cv's (table, folds) for settings {(FS, ndes): (held-out MCC, gap)}.

    Folds share a difficulty effect, as real folds do, so the paired tests can see small differences.
    """
    n = n_splits * n_repeats
    difficulty = rng.normal(0, 0.05, size=n)
    rows, fold_rows = [], []
    for (fs, ndes), (held, gap) in settings.items():
        scores = held + difficulty + rng.normal(0, 0.01, size=n)
        train = scores + gap + rng.normal(0, 0.01, size=n)
        fold_rows += [{"FS": fs, "ndes": ndes, "repeat": i // n_splits, "fold": i, "n_features": ndes,
                       "MCC": scores[i], "MCC_train": train[i]} for i in range(n)]
        rows.append({"FS": fs, "ndes": ndes, "n_features": float(ndes), "MCC": float(scores.mean())})
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def _check_choice(report):
    """What best_without_overfitting promises, whatever the data."""
    assert report["chosen"].sum() == 1
    chosen = report[report["chosen"]].iloc[0]
    assert chosen["competitive"] and not chosen["overfits"]
    eligible = report[report["competitive"] & ~report["overfits"]]
    assert chosen["MCC"] == eligible["MCC"].max()
    assert report["MCC"].is_monotonic_decreasing
    return chosen


def test_best_without_overfitting_passes_over_the_best_score_when_it_overfits(rng):
    table, folds = _synthetic_cv(rng, {("rfe_clas", 800): (0.50, 0.30),  # best score, large gap
                                       ("clas", 600): (0.495, 0.10),     # about as good, small gap
                                       ("clas", 50): (0.20, 0.05)})      # smallest gap, clearly worse
    report = best_without_overfitting(table, folds).set_index(["FS", "ndes"])
    assert not report.loc[("clas", 50), "competitive"]
    assert report.loc[("rfe_clas", 800), "overfits"]
    assert _check_choice(report.reset_index())["FS"] == "clas"
    assert report.loc[("clas", 600), "chosen"]


def test_best_without_overfitting_keeps_the_best_score_when_gaps_are_alike(rng):
    table, folds = _synthetic_cv(rng, {("rfe_clas", 800): (0.50, 0.10), ("clas", 600): (0.49, 0.10)})
    report = best_without_overfitting(table, folds)
    assert _check_choice(report)["FS"] == "rfe_clas"


def test_best_without_overfitting_n_splits_sets_the_correction(rng):
    # One repeat of many resplits: the inferred n_splits would be the number of
    # resplits; passing 5 gives the 80:20 correction, a more cautious test.
    table, folds = _synthetic_cv(rng, {("rfe_clas", 800): (0.50, 0.14), ("clas", 600): (0.495, 0.10)},
                                 n_splits=20, n_repeats=1)
    inferred = best_without_overfitting(table, folds).set_index(["FS", "ndes"])
    corrected = best_without_overfitting(table, folds, n_splits=5).set_index(["FS", "ndes"])
    assert corrected.loc[("rfe_clas", 800), "p_overfits"] > inferred.loc[("rfe_clas", 800), "p_overfits"]


def test_best_without_overfitting_max_gap_limits_the_gap_share(rng):
    # The best score overfits by 40% of its training score; the significance
    # steps alone keep it (every setting overfits alike), the limit does not.
    settings = {("clas", 800): (0.50, 0.33), ("clas", 600): (0.48, 0.32), ("clas", 100): (0.40, 0.08)}
    table, folds = _synthetic_cv(rng, settings)
    alone = best_without_overfitting(table, folds).set_index(["FS", "ndes"])
    assert not alone.loc[("clas", 100), "competitive"]
    limited = best_without_overfitting(table, folds, max_gap=0.2).set_index(["FS", "ndes"])
    assert {"gap_share", "within_max_gap"} <= set(limited.columns)
    assert not limited.loc[("clas", 800), "within_max_gap"] and not limited.loc[("clas", 800), "competitive"]
    assert limited.loc[("clas", 100), "chosen"]
    assert limited.loc[limited["chosen"], "gap_share"].iloc[0] <= 0.2


def test_best_without_overfitting_max_gap_chooses_nothing_when_nothing_qualifies(rng):
    table, folds = _synthetic_cv(rng, {("clas", 800): (0.50, 0.33), ("clas", 600): (0.48, 0.32)})
    report = best_without_overfitting(table, folds, max_gap=0.2)
    assert not report["chosen"].any() and not report["within_max_gap"].any()
    assert len(report) == 2


def test_best_without_overfitting_on_feature_selection_cv(rng):
    X, y = _binary_features(rng, n=150, informative=5, noise=40)
    configs = [("clas", 5), ("clas", 20), ("rfe_clas", 5), ("rfe_clas", 20)]
    table, folds = feature_selection_cv(X, y, configs, n_splits=3, n_repeats=3, rs=0, return_folds=True)
    report = best_without_overfitting(table, folds, metric="F2")
    assert report["chosen"].sum() == 1
    chosen = report[report["chosen"]].iloc[0]
    assert chosen["competitive"] and not chosen["overfits"]
    assert {"F2_train", "gap", "p_below_best", "p_overfits"} <= set(report.columns)
    assert report["F2"].is_monotonic_decreasing


def test_elastic_net_zeroes_out_noise_features(rng):
    # A real elastic net, whatever the scikit-learn version: before 1.8 a
    # LogisticRegression ignores l1_ratio unless the penalty is named, and an
    # L2 penalty zeroes nothing out.
    X, y = _binary_features(rng, n=150, informative=3, noise=60)
    kept = select_features(X, y, "enet_clas", None, cor=None, enet_params={"C": 0.1, "random_state": 0})
    assert 0 < len(kept) < X.shape[1]


def test_enet_params_reach_the_elastic_net(rng, cwd_tmp_path):
    X, y = _binary_features(rng, n=150, informative=3, noise=60)
    strong = select_features(X, y, "enet_clas", None, cor=None, enet_params={"C": 0.05, "random_state": 0})
    weak = select_features(X, y, "enet_clas", None, cor=None, enet_params={"C": 10, "random_state": 0})
    assert len(strong) < len(weak)
    # the same settings through data_prep
    df = X.assign(name=[f"m{i}" for i in range(len(X))], y=y)
    df.to_csv("enet.csv", index=False)
    kwargs = {"TARGET": "y", "mod": "class", "ratio": 0.2, "rs": 0, "FS": "enet_clas", "ndes": None}
    few = data_prep("enet.csv", enet_params={"C": 0.05, "random_state": 0}, **kwargs)[4]
    many = data_prep("enet.csv", enet_params={"C": 10, "random_state": 0}, **kwargs)[4]
    assert len(few) < len(many)


def test_select_features_matches_data_prep(rng, cwd_tmp_path):
    X, y = _binary_features(rng, n=150, informative=4, noise=30)
    X["copy"] = X["i0"]  # dropped by the correlation filter
    X["constant"] = 1  # dropped by the variance filter
    df = X.assign(name=[f"m{i}" for i in range(len(X))], y=y)
    df.to_csv("fs.csv", index=False)
    X_train, y_train, _, _, v_names = data_prep("fs.csv", TARGET="y", mod="class", ratio=0.2, rs=0,
                                                Cor="on", FS="clas", ndes=10)[:5]
    full = df.drop(columns=["name", "y"]).loc[X_train.index]
    assert select_features(full, y_train, "clas", 10) == list(v_names)
    kept = select_features(full, y_train)  # the filters alone
    assert "constant" not in kept and not {"copy", "i0"} <= set(kept)
    assert len(kept) == X.shape[1] - 2


def test_select_features_warns_when_the_elastic_net_does_not_converge(rng):
    from sklearn.exceptions import ConvergenceWarning

    X, y = _binary_features(rng, n=100, informative=3, noise=20)
    with pytest.warns(ConvergenceWarning):
        select_features(X, y, "enet_clas", 5, enet_params={"max_iter": 1, "random_state": 0})


def test_feature_selection_cv_compares_several_models_on_shared_selections(rng):
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import BernoulliNB, ComplementNB

    X, y = _binary_features(rng, n=150, informative=5, noise=25)
    configs = [("clas", 5), ("rfe_clas", 8), ("off", None)]
    models = {"cnb": ComplementNB(), "bnb": BernoulliNB(), "lr": LogisticRegression(max_iter=1000)}
    table, folds = feature_selection_cv(X, y, configs, model=models, n_splits=3, n_repeats=2, rs=0,
                                        return_folds=True)
    assert list(table["model"]) == [m for m in models for _ in configs]
    assert len(folds) == len(models) * len(configs) * 6
    # every model sees the same selection in every fold
    assert folds.groupby(["FS", "ndes", "fold"], dropna=False)["n_features"].nunique().eq(1).all()
    off = table[table["FS"] == "off"]
    assert off["n_features"].eq(X.shape[1]).all() and off["ndes"].isna().all()
    # a model in a dict scores as it does on its own
    alone = feature_selection_cv(X, y, configs, model=ComplementNB(), n_splits=3, n_repeats=2, rs=0)
    pd.testing.assert_frame_equal(table[table["model"] == "cnb"].drop(columns="model").reset_index(drop=True),
                                  alone)
    report = best_without_overfitting(table, folds, metric="F2")
    assert report["chosen"].sum() == 1
    assert report.groupby(["model", "FS", "ndes"], dropna=False).size().eq(1).all()


def test_feature_selection_cv_reports_pr_auc(rng):
    from sklearn.svm import LinearSVC

    X, y = _binary_features(rng, n=150, informative=5, noise=25)
    table, folds = feature_selection_cv(X, y, [("clas", 5)], model={"svm": LinearSVC()}, n_splits=3,
                                        rs=0, return_folds=True)
    assert table["PR_AUC"].between(0, 1).all()
    assert table.loc[0, "PR_AUC"] > np.mean(y) + 0.2  # well above a random ranking's
    assert {"PR_AUC", "PR_AUC_train"} <= set(folds.columns)


def test_feature_selection_cv_counts_unconverged_elastic_nets(rng):
    X, y = _binary_features(rng, n=100, informative=3, noise=20)
    configs = [("enet_clas", 5), ("enet_clas", 10), ("clas", 5)]
    table = feature_selection_cv(X, y, configs, n_splits=3, rs=0, enet_params={"max_iter": 1, "random_state": 0})
    assert list(table["unconverged"][:2]) == [3, 3] and pd.isna(table["unconverged"][2])
    plain = feature_selection_cv(X, y, [("clas", 5)], n_splits=3, rs=0)
    assert "unconverged" not in plain.columns


def test_feature_selection_cv_with_a_weak_penalty_keeps_distinct_counts(rng):
    X, y = _binary_features(rng, n=150, informative=5, noise=60)
    configs = [("enet_clas", k) for k in (5, 10, 20, 40)]
    table = feature_selection_cv(X, y, configs, n_splits=3, rs=0, enet_params={"C": 100, "random_state": 0})
    assert list(table["n_features"]) == [5, 10, 20, 40]


def test_feature_selection_cv_memory_reuses_fits(rng, tmp_path):
    X, y = _binary_features(rng, n=120, informative=4, noise=30)
    configs = [("enet_clas", 5), ("off", None)]
    params = {"C": 1, "random_state": 0}
    first = feature_selection_cv(X, y, configs, n_splits=3, rs=0, enet_params=params, memory=tmp_path)
    assert any(tmp_path.rglob("output.pkl"))  # joblib stored the fits
    again = feature_selection_cv(X, y, configs, n_splits=3, rs=0, enet_params=params, memory=tmp_path)
    pd.testing.assert_frame_equal(first, again)


def test_resplit_evaluation_scores_every_resplit(rng):
    from sklearn.naive_bayes import ComplementNB

    X, y = _binary_features(rng, n=150, informative=5, noise=25)
    scores, importances = resplit_evaluation(X, y, ComplementNB(), "clas", 8, n_resplits=12, rs=0)
    assert list(scores["resplit"]) == list(range(12)) and (scores["n_features"] == 8).all()
    assert scores[["F2", "recall", "precision", "PR_AUC", "F2_train"]].stack().between(0, 1).all()
    assert scores["MCC"].mean() > 0.3
    assert importances.shape == (12, X.shape[1])
    assert importances.notna().sum(axis=1).eq(8).all()  # NaN where a resplit did not keep a feature
    again, _ = resplit_evaluation(X, y, ComplementNB(), "clas", 8, n_resplits=12, rs=0)
    pd.testing.assert_frame_equal(scores, again)


def test_resplit_evaluation_shuffled_labels_score_at_chance(rng):
    from sklearn.naive_bayes import ComplementNB

    X, y = _binary_features(rng, n=150, informative=5, noise=25)
    real, _ = resplit_evaluation(X, y, ComplementNB(), "clas", 8, n_resplits=10, rs=0, importance=False)
    shuffled, none = resplit_evaluation(X, y, ComplementNB(), "clas", 8, n_resplits=10, rs=0,
                                        shuffle_labels=True, importance=False)
    assert none is None
    assert abs(shuffled["MCC"].mean()) < 0.2 < real["MCC"].mean()


def test_resplit_evaluation_reports_elastic_net_convergence(rng):
    from sklearn.naive_bayes import ComplementNB

    X, y = _binary_features(rng, n=100, informative=3, noise=20)
    scores, _ = resplit_evaluation(X, y, ComplementNB(), "enet_clas", 5, n_resplits=3, rs=0,
                                   enet_params={"max_iter": 1, "random_state": 0}, importance=False)
    assert not scores["converged"].any()


def test_paired_comparison_verdicts(rng):
    base = rng.normal(0.6, 0.05, size=50)
    scores = pd.DataFrame({
        "best": base + 0.10 + rng.normal(0, 0.01, 50),
        "same": base + 0.10 + rng.normal(0, 0.01, 50),
        "worse": base + rng.normal(0, 0.01, 50),
        "noisy": base + 0.10 + rng.normal(0, 0.3, 50),
    })
    table = paired_comparison(scores, reference="best", margin=0.05)
    assert table.loc["best", "verdict"] == "reference" and pd.isna(table.loc["best", "p"])
    assert table.loc["same", "verdict"] == "equivalent"
    assert table.loc["worse", "verdict"] == "worse"
    assert table.loc["noisy", "verdict"] == "inconclusive"
    assert (table["low"] <= table["difference"]).all() and (table["difference"] <= table["high"]).all()
    assert paired_comparison(scores).index[0] in ("best", "same", "noisy")  # default reference: highest mean
    # the interval matches the corrected test: 0 inside it exactly when p >= alpha
    for model in ("same", "worse", "noisy"):
        row = table.loc[model]
        assert (row["low"] <= 0 <= row["high"]) == (row["p"] >= 0.05)


def test_holdout_scores_known_counts():
    from scipy.stats import beta

    truth = np.array([1] * 30 + [0] * 170)
    predicted = np.array([1] * 24 + [0] * 6 + [1] * 26 + [0] * 144)
    table = holdout_scores(truth, predicted, rs=0)
    assert table.loc["recall", "count"] == "24/30" and table.loc["recall", "value"] == pytest.approx(0.8)
    # the exact (Clopper-Pearson) interval
    assert table.loc["recall", "low"] == pytest.approx(beta.ppf(0.025, 24, 7))
    assert table.loc["recall", "high"] == pytest.approx(beta.ppf(0.975, 25, 6))
    assert table.loc["precision", "count"] == "24/50"
    assert (table["low"] <= table["value"]).all() and (table["value"] <= table["high"]).all()
    assert table.loc["F2", "interval"] == "bootstrap" and "PR_AUC" not in table.index


def test_holdout_scores_pr_auc(rng):
    truth = np.array([1] * 30 + [0] * 170)
    scores = truth + rng.normal(0, 0.6, size=200)
    table = holdout_scores(truth, (scores > 0.5).astype(int), scores=scores, n_boot=300, rs=0)
    assert table.loc["PR_AUC", "low"] <= table.loc["PR_AUC", "value"] <= table.loc["PR_AUC", "high"]
    assert table.loc["PR_AUC", "value"] > 0.6


def test_data_prep_unknown_fs_raises(dataset_csv):
    with pytest.raises(ValueError):
        data_prep(dataset_csv, FS="bogus", mod="reg", ratio=0.25, rs=0)


def test_data_prep_ratio_zero_uses_all_data_for_train_and_test(dataset_csv):
    result = data_prep(dataset_csv, mod="reg", ratio=0, rs=0)
    X_train, y_train, X_test, y_test = result[0], result[1], result[2], result[3]
    assert len(X_train) == len(X_test) == 60
    assert len(y_train) == len(y_test) == 60


@pytest.fixture
def scaffold_smiles():
    # 4 distinct Bemis-Murcko scaffolds (benzene, pyridine, naphthalene, cyclohexane),
    # 3 differently-substituted analogs each -- Murcko scaffold extraction strips the
    # substituent chains, so each trio reduces to the same bare ring scaffold.
    names = [f"m{i}" for i in range(12)]
    smiles = [
        "c1ccccc1C(=O)O", "c1ccccc1C(=O)N", "c1ccccc1CCN",  # benzene scaffold
        "c1ccncc1C(=O)O", "c1ccncc1C(=O)N", "c1ccncc1CCN",  # pyridine scaffold
        "c1ccc2ccccc2c1C(=O)O", "c1ccc2ccccc2c1C(=O)N", "c1ccc2ccccc2c1CCN",  # naphthalene
        "C1CCCCC1C(=O)O", "C1CCCCC1C(=O)N", "C1CCCCC1CCN",  # cyclohexane
    ]
    groups = {names[i]: i // 3 for i in range(12)}  # which of the 4 scaffolds each name belongs to
    return names, smiles, groups


def test_scaffold_split_keeps_same_scaffold_together(scaffold_smiles):
    names, smiles, groups = scaffold_smiles
    train_names, test_names = scaffold_split(names, smiles, test_size=0.5, rs=0)

    assert set(train_names) | set(test_names) == set(names)
    assert set(train_names).isdisjoint(test_names)

    train_scaffold_ids = {groups[n] for n in train_names}
    test_scaffold_ids = {groups[n] for n in test_names}
    assert train_scaffold_ids.isdisjoint(test_scaffold_ids)


def test_scaffold_split_invalid_smiles_does_not_raise():
    names = ["good1", "good2", "bad"]
    smiles = ["c1ccccc1C(=O)O", "c1ccccc1CCN", "not-a-smiles"]
    train_names, test_names = scaffold_split(names, smiles, test_size=0.34, rs=0)
    assert set(train_names) | set(test_names) == set(names)
