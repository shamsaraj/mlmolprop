"""Tests for mlmolprop.model -- parametrized across every supported model type."""

import math
import sys

import numpy as np
import pandas as pd
import pytest

from mlmolprop.model import (
    Model,
    ModelC,
    clus_uns,
    metr,
    plot_confusion_matrix,
    plot_w,
    rocc,
    safe_divide,
)

REGRESSORS = [
    "pls",
    "mlr",
    "rf",
    "svm",
    "lsvm",
    "lasso",
    "nn",
    "tree",
    "rg",
    "el",
    "la",
    "ll",
    "or",
    "brg",
    "ardr",
    "ransa",
    "the",
    "hub",
    "sgdr",
    "kn",
    "gu",
    "ex",
    "bg",
    "gb",
    "hgb",
    "xgb",
    "ada",
]

CLASSIFIERS = [
    "tree",
    "nn",
    "rf",
    "ex",
    "lsvm",
    "svm",
    "lr",
    "ld",
    "rg",
    "per",
    "pass",
    "qua",
    "sgdc",
    "kn",
    "rn",
    "gu",
    "gunb",
    "cnb",
    "bg",
    "gb",
    "hgb",
    "xgb",
    "ada",
]


@pytest.fixture
def reg_train_test(regression_xy, cwd_tmp_path):
    X, y = regression_xy
    return X.iloc[:22], y.iloc[:22], X.iloc[22:], y.iloc[22:], list(X.columns)


@pytest.fixture
def clas_train_test(rng, cwd_tmp_path):
    # non-negative, scaled features so ComplementNB/RadiusNeighborsClassifier
    # (both sensitive to feature scale/sign) work like they would on real
    # scaled QSAR descriptors
    n = 40
    X = pd.DataFrame(rng.uniform(0, 1, size=(n, 4)), columns=["f1", "f2", "f3", "f4"])
    score = X["f1"] * 2 - X["f2"]
    y = pd.Series((score > np.median(score)).astype(int))
    return X.iloc[:30], y.iloc[:30], X.iloc[30:], y.iloc[30:], list(X.columns)


# Test data is tiny (a handful of rows, 3-4 features), and each model's
# hyperparameters now live under real sklearn names rather than an
# overloaded `c`/`c1` slot -- most model types never actually read that
# slot in the original code either (only pls/rf/svm/tree did for
# regression; more did for classification), so only the ones that need a
# small value to fit this dataset without erroring or running slowly get
# an entry here.
REGRESSOR_TEST_PARAMS = {
    "pls": {"n_components": 3},  # n_components must be <= n_features (3)
    "rf": {"n_estimators": 3, "max_depth": 2, "max_features": 3},
    "svm": {"C": 3},
    "tree": {"max_depth": 3, "max_features": 3},
    # default min_samples_leaf=20 would barely allow any real split on this
    # fixture's 22-row train set.
    "hgb": {"max_iter": 5, "max_depth": 2, "min_samples_leaf": 2},
    "xgb": {"n_estimators": 3, "max_depth": 2},
}

CLASSIFIER_TEST_PARAMS = {
    "tree": {"max_depth": 3},
    "nn": {"hidden_layer_sizes": (3, 3, 3)},
    "rf": {"max_depth": 3, "n_estimators": 6},
    "ex": {"max_depth": 3},
    "lsvm": {"C": 3},
    "svm": {"C": 3},
    "lr": {"max_iter": 3},
    "kn": {"n_neighbors": 3},
    "cnb": {"alpha": 3},
    "hgb": {"max_iter": 5, "max_depth": 2, "min_samples_leaf": 2},
    "xgb": {"n_estimators": 3, "max_depth": 2},
}


@pytest.mark.parametrize("M", REGRESSORS)
def test_model_every_regressor_type(reg_train_test, M):
    X_train, y_train, X_test, y_test, v_names = reg_train_test
    result, cv_metrics, fitted_model, analysis = Model(
        X_train,
        y_train,
        X_test,
        y_test,
        v_names,
        params=REGRESSOR_TEST_PARAMS.get(M, {}),
        M=M,
        rs=0,
        cv="loo",
    )
    assert "R2" in result
    assert cv_metrics is not None
    assert analysis is not None


def test_model_reports_nan_f_instead_of_raising_when_features_exceed_rows(rng, cwd_tmp_path):
    # regression guard: F()'s n > k+1 requirement is a classical OLS concept
    # that doesn't mean anything for RF's capacity -- Model() used to
    # propagate F()'s ValueError unconditionally for every M=, blocking any
    # regressor (not just linear ones) from running when features >= rows.
    n = 10
    X = pd.DataFrame(rng.normal(size=(n, 15)), columns=[f"f{i}" for i in range(15)])
    y = pd.Series(rng.normal(size=n))
    X_train, y_train, X_test, y_test = X.iloc[:8], y.iloc[:8], X.iloc[8:], y.iloc[8:]

    result, _, _, _ = Model(
        X_train, y_train, X_test, y_test, list(X.columns),
        params={"n_estimators": 3, "max_depth": 2, "max_features": 3},
        M="rf", rs=0, cv="off",
    )
    assert math.isnan(result["F"])
    assert result["R2"] is not None and not math.isnan(result["R2"])


@pytest.mark.parametrize("M", ["mlr", "rf", "pls"])
def test_model_f_statistic_is_never_hardcoded_zero(reg_train_test, M):
    # Property: a real F-statistic landing on exactly 0.0 is vanishingly
    # unlikely across different model types/data -- "always identically 0"
    # is itself the signal that it's not being computed at all.
    X_train, y_train, X_test, y_test, v_names = reg_train_test
    result, _, _, _ = Model(
        X_train, y_train, X_test, y_test, v_names,
        params=REGRESSOR_TEST_PARAMS.get(M, {}), M=M, rs=0, cv="loo",
    )
    assert result["F"] != 0.0


def test_model_f_statistic_near_null_value_for_unrelated_data(rng, cwd_tmp_path):
    # Edge case: y with essentially zero real relationship to x. A hardcoded
    # 0.0 happens to be numerically plausible near the null-hypothesis
    # region, so a loose "is F near 1" tolerance wouldn't actually
    # distinguish the bug from a fix -- compare against the independently
    # computed value instead, which does. (0.59 here, not 0 -- verified by
    # direct computation, not assumed.)
    from mlmolprop.basic import F as basic_F

    n = 30
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f1", "f2", "f3"])
    y = pd.Series(rng.normal(size=n))  # unrelated to X by construction
    X_train, y_train, X_test, y_test = X.iloc[:22], y.iloc[:22], X.iloc[22:], y.iloc[22:]

    result, _, fitted_model, _ = Model(
        X_train, y_train, X_test, y_test, list(X.columns), M="mlr", rs=0, cv="loo",
    )
    y_predict_train = fitted_model.predict(X_train)
    expected_f = basic_F(y_train, y_predict_train, k=3)
    assert result["F"] == pytest.approx(expected_f)


def test_model_f_statistic_known_answer_matches_basic_F(reg_train_test):
    # Known-answer case: the correct value is directly computable via
    # basic.F(), which already exists and is independently tested
    # (test_basic.py) -- this just checks Model() is wired to use it.
    from mlmolprop.basic import F as basic_F

    X_train, y_train, X_test, y_test, v_names = reg_train_test
    result, _, fitted_model, _ = Model(
        X_train, y_train, X_test, y_test, v_names, M="mlr", rs=0, cv="loo",
    )
    y_predict_train = fitted_model.predict(X_train)
    expected_f = basic_F(y_train, y_predict_train, k=len(v_names))
    assert result["F"] == pytest.approx(expected_f)


def test_model_nn_variable_importance_has_one_value_per_feature(reg_train_test):
    # Property: "Variable Importance" should have exactly one column (one
    # value per feature), not one column per unit in the second hidden
    # layer.
    X_train, y_train, X_test, y_test, v_names = reg_train_test
    result, _, _, _ = Model(
        X_train, y_train, X_test, y_test, v_names, M="nn", rs=0, cv="off",
        params={"max_iter": 50},  # default hidden_layer_sizes: 3 hidden layers
    )
    assert result["Variable Importance"].shape == (len(v_names), 1)


def test_model_nn_variable_importance_correct_for_single_hidden_layer(reg_train_test):
    # Edge case: with only ONE hidden layer, coefs_ has exactly 2 matrices,
    # so the current np.dot(coefs_[0], coefs_[1]) genuinely covers the
    # whole network -- confirms the bug is specifically about >=2 hidden
    # layers, not the approach being wrong in general.
    X_train, y_train, X_test, y_test, v_names = reg_train_test
    result, _, _, _ = Model(
        X_train, y_train, X_test, y_test, v_names, M="nn", rs=0, cv="off",
        params={"max_iter": 50, "hidden_layer_sizes": (5,)},
    )
    assert result["Variable Importance"].shape == (len(v_names), 1)


def test_model_nn_variable_importance_ranks_dominant_feature_highest(rng, cwd_tmp_path):
    # Known-answer case (directional -- there's no simple closed form for
    # NN importance in general), using the same single-hidden-layer config
    # as the edge case above (the one config the current shape isn't wrong
    # for): one feature obviously dominates the target, so it should be
    # ranked most important. Deliberately *not* testing this against the
    # buggy multi-layer default -- verified directly that the wrong-shaped
    # VI's column 0 doesn't reliably break the ranking either way, so an
    # xfail there would be flaky rather than a real demonstration; the
    # property test above already covers the shape bug unconditionally.
    # Scaled features, enough data/iterations to actually converge (a first
    # attempt with unscaled features and few iterations produced an
    # unconverged fit -- ConvergenceWarning -- and a genuinely unreliable
    # ranking; this config gets R^2 > 0.99 on train, a real fit).
    n = 200
    X = pd.DataFrame(rng.uniform(-1, 1, size=(n, 3)), columns=["dominant", "f2", "f3"])
    y = pd.Series(10 * X["dominant"] + 0.01 * rng.normal(size=n))
    X_train, y_train, X_test, y_test = X.iloc[:150], y.iloc[:150], X.iloc[150:], y.iloc[150:]

    result, _, _, _ = Model(
        X_train, y_train, X_test, y_test, list(X.columns), M="nn", rs=0, cv="off",
        params={"max_iter": 2000, "hidden_layer_sizes": (5,)},
    )
    vi = result["Variable Importance"]
    most_important = vi.abs().iloc[:, 0].idxmax()
    assert most_important == "dominant"


@pytest.mark.parametrize("M", CLASSIFIERS)
def test_modelc_every_classifier_type(clas_train_test, M):
    X_train, y_train, X_test, y_test, v_names = clas_train_test
    result, fitted_model = ModelC(
        X_train,
        y_train,
        X_test,
        y_test,
        v_names,
        params=CLASSIFIER_TEST_PARAMS.get(M, {}),
        M=M,
        rs=0,
        cv="loo",
    )
    assert 0.0 <= result["accuracy_score_train"] <= 1.0
    assert 0.0 <= result["accuracy_score_test"] <= 1.0


def test_model_cv_off_does_not_crash_and_matches_cv_on(reg_train_test):
    # regression guard: `List` (CV metrics) and `analysis` were only assigned
    # inside `if cv != "off":` but returned unconditionally, so cv="off"
    # always raised NameError
    X_train, y_train, X_test, y_test, v_names = reg_train_test
    rf_params = {"n_estimators": 3, "max_depth": 2, "max_features": 3}
    r_off = Model(
        X_train,
        y_train,
        X_test,
        y_test,
        v_names,
        params=rf_params,
        M="rf",
        rs=0,
        cv="off",
    )
    r_loo = Model(
        X_train,
        y_train,
        X_test,
        y_test,
        v_names,
        params=rf_params,
        M="rf",
        rs=0,
        cv="loo",
    )

    assert r_off[1] is None
    assert r_off[3] is None
    assert r_loo[1] is not None
    # the CV loop refits a *separate* model object, so it must have zero
    # effect on the officially reported model's own metrics
    assert r_off[0]["R2"] == r_loo[0]["R2"]
    assert r_off[0]["R2_test"] == r_loo[0]["R2_test"]


def test_model_path_param_writes_into_given_directory(reg_train_test, cwd_tmp_path):
    # regression guard: train.csv/test.csv were hardcoded to the caller's
    # CWD; `path` (matching clus_uns()'s existing "directory prefix"
    # convention) should redirect them instead of leaving them at CWD.
    X_train, y_train, X_test, y_test, v_names = reg_train_test
    out_dir = cwd_tmp_path / "run1"
    out_dir.mkdir()
    Model(
        X_train, y_train, X_test, y_test, v_names,
        params={"n_estimators": 3, "max_depth": 2, "max_features": 3},
        M="rf", rs=0, cv="off", path=str(out_dir) + "/",
    )
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "test.csv").exists()
    assert not (cwd_tmp_path / "train.csv").exists()
    assert not (cwd_tmp_path / "test.csv").exists()


def test_modelc_path_param_writes_into_given_directory(clas_train_test, cwd_tmp_path):
    X_train, y_train, X_test, y_test, v_names = clas_train_test
    out_dir = cwd_tmp_path / "run1"
    out_dir.mkdir()
    ModelC(
        X_train, y_train, X_test, y_test, v_names,
        params={"max_depth": 3, "n_estimators": 6},
        M="rf", rs=0, cv="kf", path=str(out_dir) + "/",
    )
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "test.csv").exists()
    assert not (cwd_tmp_path / "train.csv").exists()
    assert not (cwd_tmp_path / "test.csv").exists()


def test_modelc_cv_off_does_not_crash_and_leaves_cv_metrics_none(clas_train_test):
    # regression guard: unlike Model(), none of ModelC's loo/kf/kfr/shuff
    # branches matched "off", so `loo` was never assigned and loo.split(x)
    # raised UnboundLocalError as soon as cv="off" was passed.
    X_train, y_train, X_test, y_test, v_names = clas_train_test
    result, fitted_model = ModelC(
        X_train, y_train, X_test, y_test, v_names,
        params={"max_depth": 3, "n_estimators": 6}, M="rf", rs=0, cv="off",
    )
    assert result["CV_metrics"] is None
    assert result["confusion matrix_CV"] is None
    assert result["accuracy_score_LOO"] == ""
    # the officially reported model's own train/test metrics must be
    # unaffected by skipping CV entirely
    assert 0.0 <= result["test_AC"] <= 1.0
    assert -1.0 <= result["test_MCC"] <= 1.0


def test_modelc_cv_off_vs_cv_kf_report_same_train_test_metrics(clas_train_test):
    # Property: CV is a separate diagnostic refit (model2 in the source) --
    # skipping it must not change the officially reported model's own
    # train/test metrics, mirroring test_model_cv_off_does_not_crash_and_matches_cv_on.
    X_train, y_train, X_test, y_test, v_names = clas_train_test
    rf_params = {"max_depth": 3, "n_estimators": 6}
    r_off, _ = ModelC(
        X_train, y_train, X_test, y_test, v_names,
        params=rf_params, M="rf", rs=0, cv="off",
    )
    r_kf, _ = ModelC(
        X_train, y_train, X_test, y_test, v_names,
        params=rf_params, M="rf", rs=0, cv="kf",
    )
    assert r_off["test_MCC"] == r_kf["test_MCC"]
    assert r_off["test_AC"] == r_kf["test_AC"]


def test_model_invalid_name_raises_value_error(reg_train_test):
    X_train, y_train, X_test, y_test, v_names = reg_train_test
    with pytest.raises(ValueError):
        Model(X_train, y_train, X_test, y_test, v_names, M="bogus")


def test_modelc_invalid_name_raises_value_error(clas_train_test):
    X_train, y_train, X_test, y_test, v_names = clas_train_test
    with pytest.raises(ValueError):
        ModelC(X_train, y_train, X_test, y_test, v_names, M="bogus")


@pytest.mark.slow
def test_modelc_dl_actually_trains(clas_train_test):
    # regression guard: the early-stopping callback was never attached to
    # any .fit() call, so epochs=early_stopping_monitor.stopped_epoch was
    # always 0 -- the model was returned untrained, with weights identical
    # to a freshly-constructed, never-trained model. Compare against a
    # same-seed fresh model rather than asserting an accuracy threshold,
    # since accuracy on a tiny stochastic net is inherently noisy run to
    # run (unlike Keras's own weight init, which is fully determined by
    # the seed) and isn't what this bug actually affected.
    keras = pytest.importorskip("keras")

    from mlmolprop.model import _build_dl_model

    X_train, y_train, X_test, y_test, v_names = clas_train_test

    keras.utils.set_random_seed(0)
    fresh_model = _build_dl_model(
        len(v_names),
        hidden_layer_sizes=[8, 4],
        dropout=0.2,
        optimizer="adam",
        learning_rate=0.01,
        nesterov=True,
    )
    fresh_weights = fresh_model.get_weights()[0].copy()

    keras.utils.set_random_seed(0)
    result, fitted_model = ModelC(
        X_train,
        y_train,
        X_test,
        y_test,
        v_names,
        M="dl",
        rs=0,
        params={"epochs": 20, "hidden_layer_sizes": [8, 4], "batch_size": 8},
    )
    trained_weights = fitted_model.get_weights()[0]

    assert not np.allclose(fresh_weights, trained_weights)
    # loose sanity check: a trained binary classifier shouldn't be worse
    # than a coin flip on a balanced task
    assert result["accuracy_score_train"] >= 0.5


def test_modelc_dl_missing_keras_raises_informative_error(monkeypatch, clas_train_test):
    # keras/torch live behind the optional "dl" extra; if they're not
    # installed, M="dl" should fail with a clear message pointing at how to
    # get them, not a bare ImportError from deep inside _build_dl_model.
    X_train, y_train, X_test, y_test, v_names = clas_train_test
    monkeypatch.setitem(sys.modules, "keras", None)

    with pytest.raises(ImportError, match=r"pip install 'mlmolprop\[dl\]'"):
        ModelC(X_train, y_train, X_test, y_test, v_names, M="dl", rs=0)


@pytest.mark.slow
def test_model_dl_actually_trains(reg_train_test):
    # Same regression guard as test_modelc_dl_actually_trains: compare
    # against a same-seed fresh (untrained) model rather than an error
    # threshold, since a tiny stochastic net's accuracy is noisy run to run
    # in a way its weights (fully determined by the seed before any
    # training) aren't.
    keras = pytest.importorskip("keras")

    from mlmolprop.model import _build_dl_model

    X_train, y_train, X_test, y_test, v_names = reg_train_test

    keras.utils.set_random_seed(0)
    fresh_model = _build_dl_model(
        len(v_names),
        hidden_layer_sizes=[8, 4],
        dropout=0.2,
        optimizer="adam",
        learning_rate=0.01,
        nesterov=True,
        task="regression",
    )
    fresh_weights = fresh_model.get_weights()[0].copy()

    keras.utils.set_random_seed(0)
    result, cv_metrics, fitted_model, analysis = Model(
        X_train,
        y_train,
        X_test,
        y_test,
        v_names,
        M="dl",
        rs=0,
        cv="kf",
        params={"epochs": 20, "hidden_layer_sizes": [8, 4], "batch_size": 8},
    )
    trained_weights = fitted_model.get_weights()[0]

    assert not np.allclose(fresh_weights, trained_weights)
    assert "R2" in result
    # M="dl" always runs its own internal 5-fold CV regardless of `cv`, but
    # only populates List/analysis when cv != "off".
    assert cv_metrics is not None
    assert analysis is not None


def test_model_dl_missing_keras_raises_informative_error(monkeypatch, reg_train_test):
    # keras/torch live behind the optional "dl" extra; if they're not
    # installed, M="dl" should fail with a clear message pointing at how to
    # get them, not a bare ImportError from deep inside _build_dl_model.
    X_train, y_train, X_test, y_test, v_names = reg_train_test
    monkeypatch.setitem(sys.modules, "keras", None)

    with pytest.raises(ImportError, match=r"pip install 'mlmolprop\[dl\]'"):
        Model(X_train, y_train, X_test, y_test, v_names, M="dl", rs=0)


def test_model_xgb_missing_xgboost_raises_informative_error(monkeypatch, reg_train_test):
    # xgboost lives behind the optional "xgboost" extra; if it's not
    # installed, M="xgb" should fail with a clear message pointing at how
    # to get it, not a bare ImportError from deep inside Model().
    X_train, y_train, X_test, y_test, v_names = reg_train_test
    monkeypatch.setitem(sys.modules, "xgboost", None)

    with pytest.raises(ImportError, match=r"pip install 'mlmolprop\[xgboost\]'"):
        Model(X_train, y_train, X_test, y_test, v_names, M="xgb", rs=0)


def test_modelc_xgb_missing_xgboost_raises_informative_error(monkeypatch, clas_train_test):
    X_train, y_train, X_test, y_test, v_names = clas_train_test
    monkeypatch.setitem(sys.modules, "xgboost", None)

    with pytest.raises(ImportError, match=r"pip install 'mlmolprop\[xgboost\]'"):
        ModelC(X_train, y_train, X_test, y_test, v_names, M="xgb", rs=0)


def test_clus_uns_pca_then_kmeans_then_pc_km_in_one_process(reg_train_test):
    # regression guard: clus_uns() never called plt.figure() between plots,
    # so M="pca" could leave a stale 3D axes as matplotlib's "current axes"
    # and break a later M="kmeans" call's 2D plt.scatter()
    X_train, y_train, X_test, y_test, v_names = reg_train_test
    y_binary = (y_train > y_train.median()).astype(int).values
    ytest_binary = (y_test > y_test.median()).astype(int).values

    clus_uns(X_train, y_binary, path="./", M="pca", n=2)
    clus_uns(
        X_train,
        y_binary,
        path="./",
        M="kmeans",
        xtest=X_test,
        ytest=ytest_binary,
        n=2,
        n2=2,
        v_names=v_names,
        v1=v_names[0],
        v2=v_names[1],
    )
    clus_uns(X_train, y_binary, path="./", M="pc-km", n=2, n2=2)


def test_clus_uns_pc_km_supports_n2_different_from_n(reg_train_test):
    # regression guard: colors3 (used for KMeans cluster centers) was sized
    # by `n` instead of `n2`, and M="kmeans"'s cluster-center scatter used a
    # hardcoded 2-color list -- both crashed with a mismatched-length error
    # from matplotlib as soon as n2 != n (or n2 != 2 for "kmeans"), e.g.
    # n=2 with n2=3 clusters.
    X_train, y_train, X_test, y_test, v_names = reg_train_test
    y_binary = (y_train > y_train.median()).astype(int).values
    clus_uns(X_train, y_binary, path="./", M="pc-km", n=2, n2=3)


def test_rocc_uses_actual_label_param_not_hardcoded(rng):
    # regression guard: EF2/EF10/EF20/EF50 hardcoded l="b" instead of using
    # the actual `l` parameter (only EF1 used it correctly)
    y = pd.Series(rng.choice([0, 1], size=30))
    x = pd.Series(rng.uniform(size=30))
    rocc(x, y, l=1)  # must not raise for a label that actually appears in y


def _parse_ef(captured_stdout: str) -> dict:
    """Pull {"EF1": value, "EF2": value, ...} out of rocc()'s printed lines."""
    ef = {}
    for line in captured_stdout.splitlines():
        if "-->" in line:
            value_str, name = line.split("-->")
            ef[name.strip()] = float(value_str.strip())
    return ef


def test_rocc_enrichment_factor_is_invariant_to_row_order(rng, capsys):
    # Property: EF is defined in terms of (score, label) pairs, not row
    # position, so permuting the rows together must not change the printed
    # EF values (the AUC/ROC part already satisfies this, via sklearn).
    n = 50
    x = pd.Series(rng.uniform(size=n))
    y = pd.Series(rng.choice([0, 1], size=n))

    rocc(x, y, l=1)
    ef_original = _parse_ef(capsys.readouterr().out)

    perm = rng.permutation(n)
    x_shuffled = x.iloc[perm].reset_index(drop=True)
    y_shuffled = y.iloc[perm].reset_index(drop=True)
    rocc(x_shuffled, y_shuffled, l=1)
    ef_shuffled = _parse_ef(capsys.readouterr().out)

    assert ef_original == ef_shuffled


def test_rocc_enrichment_factor_not_contradicted_by_perfect_auc(capsys):
    # Edge case: actives have the 10 highest scores (perfect separation,
    # AUC=1.0) but are placed at the END of y's positional order, so a
    # score-blind, position-based EF calculation gets them completely
    # backwards.
    x = pd.Series(np.concatenate([np.linspace(5, 0, 90), np.linspace(10, 9, 10)]))
    y = pd.Series([0] * 90 + [1] * 10)

    rocc(x, y, l=1)
    out = capsys.readouterr().out
    ef = _parse_ef(out)

    assert "1.0" in out  # AUC: perfect separation
    assert ef["EF10"] > 0  # top-10%-by-score are the actives -- should enrich


def test_rocc_enrichment_factor_known_answer_when_input_is_prescored(capsys):
    # Known-answer case: same scenario as the perfect-AUC edge case above,
    # but with y already given in score-sorted order (actives first) --
    # the precondition COUNT() silently assumes. Confirms the EF *formula*
    # itself is correct; only the missing sort is the bug. Hand-computed:
    # baseline=10/100=0.1; top 1%/2%/10% are all-active -> rate=1.0 ->
    # EF=1.0/0.1=10; top 20% has 10 actives of 20 -> rate=0.5 -> EF=5;
    # top 50% has 10 actives of 50 -> rate=0.2 -> EF=2.
    x = pd.Series(np.concatenate([np.linspace(10, 9, 10), np.linspace(5, 0, 90)]))
    y = pd.Series([1] * 10 + [0] * 90)

    rocc(x, y, l=1)
    ef = _parse_ef(capsys.readouterr().out)

    assert ef == {
        "EF1": pytest.approx(10.0),
        "EF2": pytest.approx(10.0),
        "EF10": pytest.approx(10.0),
        "EF20": pytest.approx(5.0),
        "EF50": pytest.approx(2.0),
    }


def test_metr_matches_hand_computed_ground_truth():
    # TP=8, TN=6, FP=4, FN=2 -> SEN=8/10=0.8, SPEC=6/10=0.6, PREC=8/12=2/3,
    # AC=(8+6)/20=0.7. Regression guard for the confusion-matrix ravel()
    # unpacking bug: mlmolprop used to report SEN=0.75, SPEC=2/3, PREC=0.6
    # for this exact case (sensitivity and precision/specificity swapped),
    # verified against these exact numbers before the fix.
    result = metr(tp1=8, tn1=6, fp1=4, fn1=2)
    assert result["AC"] == pytest.approx(0.7)
    assert result["SEN"] == pytest.approx(0.8)
    assert result["SPEC"] == pytest.approx(0.6)
    assert result["PREC"] == pytest.approx(2 / 3)
    assert set(result.keys()) == {"AC", "SEN", "SPEC", "PREC", "F", "MCC"}


def test_confusion_matrix_ravel_order_assumption():
    # Documents/locks in the sklearn convention every confusion-matrix
    # unpacking in model.py depends on: for binary labels [0, 1],
    # confusion_matrix(...).ravel() is [tn, fp, fn, tp], NOT [tp, fp, fn, tn].
    from sklearn.metrics import confusion_matrix

    y_true = [1] * 10 + [0] * 10
    y_pred = [1] * 8 + [0] * 2 + [1] * 4 + [0] * 6  # TP=8, FN=2, FP=4, TN=6
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    assert (tn, fp, fn, tp) == (6, 4, 2, 8)


def test_modelc_metrics_match_sklearn_reference(clas_train_test):
    # Cross-check ModelC()'s reported train-set sensitivity/specificity/
    # precision against sklearn's own recall_score/precision_score on the
    # same predictions, rather than hand-computed numbers -- an
    # independent reference for the confusion-matrix ravel() fix.
    from sklearn.metrics import precision_score, recall_score

    X_train, y_train, X_test, y_test, v_names = clas_train_test
    result, fitted = ModelC(
        X_train,
        y_train,
        X_test,
        y_test,
        v_names,
        M="rf",
        params={"max_depth": 3, "n_estimators": 6},
        rs=0,
        cv="kf",
    )

    y_pred_train = fitted.predict(X_train)
    expected_sen = recall_score(y_train, y_pred_train, pos_label=1)
    expected_spec = recall_score(y_train, y_pred_train, pos_label=0)
    expected_prec = precision_score(y_train, y_pred_train, pos_label=1)

    assert result["train_metrics"]["SEN"] == pytest.approx(expected_sen)
    assert result["train_metrics"]["SPEC"] == pytest.approx(expected_spec)
    assert result["train_metrics"]["PREC"] == pytest.approx(expected_prec)


def test_safe_divide_by_zero_returns_zero():
    assert safe_divide(1, 0) == 0
    assert safe_divide(10, 5) == 2


def test_plot_w_and_plot_confusion_matrix_do_not_raise():
    plot_w(["a", "b"], [1, 2])
    plot_confusion_matrix(np.array([[5, 1], [2, 7]]), classes=["a", "b"])
