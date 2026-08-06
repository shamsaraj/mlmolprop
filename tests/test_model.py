"""Tests for mlmolprop.model -- parametrized across every supported model type."""

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


@pytest.mark.parametrize("M", REGRESSORS)
def test_model_every_regressor_type(reg_train_test, M):
    X_train, y_train, X_test, y_test, v_names = reg_train_test
    result, cv_metrics, fitted_model, analysis = Model(
        X_train, y_train, X_test, y_test, v_names, c=3, M=M, rs=0, cv="loo"
    )
    assert "R2" in result
    assert cv_metrics is not None
    assert analysis is not None


@pytest.mark.parametrize("M", CLASSIFIERS)
def test_modelc_every_classifier_type(clas_train_test, M):
    X_train, y_train, X_test, y_test, v_names = clas_train_test
    result, fitted_model = ModelC(
        X_train, y_train, X_test, y_test, v_names, c1=3, M=M, rs=0, cv="loo"
    )
    assert 0.0 <= result["accuracy_score_train"] <= 1.0
    assert 0.0 <= result["accuracy_score_test"] <= 1.0


def test_model_cv_off_does_not_crash_and_matches_cv_on(reg_train_test):
    # regression guard: `List` (CV metrics) and `analysis` were only assigned
    # inside `if cv != "off":` but returned unconditionally, so cv="off"
    # always raised NameError
    X_train, y_train, X_test, y_test, v_names = reg_train_test
    r_off = Model(
        X_train, y_train, X_test, y_test, v_names, c=3, M="rf", rs=0, cv="off"
    )
    r_loo = Model(
        X_train, y_train, X_test, y_test, v_names, c=3, M="rf", rs=0, cv="loo"
    )

    assert r_off[1] is None
    assert r_off[3] is None
    assert r_loo[1] is not None
    # the CV loop refits a *separate* model object, so it must have zero
    # effect on the officially reported model's own metrics
    assert r_off[0]["R2"] == r_loo[0]["R2"]
    assert r_off[0]["R2_test"] == r_loo[0]["R2_test"]


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
    import keras

    from mlmolprop.model import _build_dl_model

    X_train, y_train, X_test, y_test, v_names = clas_train_test

    keras.utils.set_random_seed(0)
    fresh_model = _build_dl_model(
        len(v_names), [8, 4], dp=0.2, omp="adam", lr1=0.01, nesterov=True
    )
    fresh_weights = fresh_model.get_weights()[0].copy()

    keras.utils.set_random_seed(0)
    result, fitted_model = ModelC(
        X_train, y_train, X_test, y_test, v_names, M="dl", rs=0, ep=20, dl2=[8, 4], bs=8
    )
    trained_weights = fitted_model.get_weights()[0]

    assert not np.allclose(fresh_weights, trained_weights)
    # loose sanity check: a trained binary classifier shouldn't be worse
    # than a coin flip on a balanced task
    assert result["accuracy_score_train"] >= 0.5


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


def test_rocc_uses_actual_label_param_not_hardcoded(rng):
    # regression guard: EF2/EF10/EF20/EF50 hardcoded l="b" instead of using
    # the actual `l` parameter (only EF1 used it correctly)
    y = pd.Series(rng.choice([0, 1], size=30))
    x = pd.Series(rng.uniform(size=30))
    rocc(x, y, l=1)  # must not raise for a label that actually appears in y


def test_metr_matthews_correlation_coefficient():
    result = metr(tp1=10, tn1=8, fp1=2, fn1=1)
    assert set(result.keys()) == {"AC", "SEN", "SPEC", "PREC", "F", "MCC"}
    assert 0 <= result["AC"] <= 1


def test_safe_divide_by_zero_returns_zero():
    assert safe_divide(1, 0) == 0
    assert safe_divide(10, 5) == 2


def test_plot_w_and_plot_confusion_matrix_do_not_raise():
    plot_w(["a", "b"], [1, 2])
    plot_confusion_matrix(np.array([[5, 1], [2, 7]]), classes=["a", "b"])
