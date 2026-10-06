"""Tests for mlmolprop.importance."""

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Perceptron,
    RANSACRegressor,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from mlmolprop.importance import (
    bootstrap_importance,
    consensus_features,
    enrichment_importance,
    lime_explain,
    partial,
    significance_importance,
    stable_features,
    variable_importance,
)


def test_lime_explain_regression_mode(regression_xy):
    from sklearn.ensemble import RandomForestRegressor

    X, y = regression_xy
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    results = lime_explain(
        model, X, list(X.columns), y.values, mode="regression", start=len(X) - 3
    )
    assert results[: len(X) - 3] == ["null"] * (len(X) - 3)
    assert all(r != "null" for r in results[len(X) - 3 :])


def test_lime_explain_invalid_mode_raises(regression_xy):
    from sklearn.ensemble import RandomForestRegressor

    X, y = regression_xy
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        lime_explain(model, X, list(X.columns), y.values, mode="bogus")


def test_partial_dependence_returns_display(regression_xy, cwd_tmp_path):
    df, y = regression_xy
    feature_importance = pd.DataFrame({0: [0.4, 0.3, 0.2]}, index=list(df.columns))
    display = partial(df, y, feature_importance, n_features=2, kind="reg", show=False)
    assert type(display).__name__ == "PartialDependenceDisplay"


def test_partial_invalid_kind_raises(regression_xy):
    df, y = regression_xy
    feature_importance = pd.DataFrame({0: [0.4, 0.3, 0.2]}, index=list(df.columns))
    with pytest.raises(ValueError):
        partial(df, y, feature_importance, kind="bogus", show=False)


@pytest.fixture
def five_feature_dataset(rng):
    n = 60
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=["f1", "f2", "f3", "f4", "f5"])
    y = pd.Series(rng.normal(size=n))
    return X, y


def _spy_selected_features(monkeypatch):
    """Capture the actual `features=` list PartialDependenceDisplay.from_estimator gets."""
    from sklearn.inspection import PartialDependenceDisplay

    captured = {}
    original = PartialDependenceDisplay.from_estimator

    def spy(estimator, X, features, feature_names=None, **kw):
        captured["features"] = list(features)
        return original(estimator, X, features=features, feature_names=feature_names, **kw)

    monkeypatch.setattr(PartialDependenceDisplay, "from_estimator", spy)
    return captured


def test_partial_plots_exactly_n_features(monkeypatch, five_feature_dataset):
    # Property: the number of features actually plotted must equal
    # n_features, for any n_features less than the total available.
    X, y = five_feature_dataset
    captured = _spy_selected_features(monkeypatch)
    feature_importance = pd.DataFrame({0: [0.4, 0.3, 0.2, 0.1, 0.05]}, index=X.columns)

    partial(X, y, feature_importance, n_features=3, kind="reg", show=False)
    assert len(captured["features"]) == 3


def test_partial_n_features_equal_to_total_does_not_overflow(monkeypatch, five_feature_dataset):
    # Edge case: n_features equal to the total number of available
    # features. Python's slicing is forgiving of an out-of-range end index
    # ([:n+1] on a 5-element list with n=5 just returns all 5, not 6), so
    # this boundary case happens to come out correct despite the bug --
    # confirms the failure is specifically "off by one when n_features <
    # total", not a crash or an unconditional overflow.
    X, y = five_feature_dataset
    captured = _spy_selected_features(monkeypatch)
    feature_importance = pd.DataFrame({0: [0.4, 0.3, 0.2, 0.1, 0.05]}, index=X.columns)

    partial(X, y, feature_importance, n_features=5, kind="reg", show=False)
    assert len(captured["features"]) == 5


def test_partial_known_answer_selects_top_ranked_features(monkeypatch, five_feature_dataset):
    # Known-answer case: 5 features with distinct importance scores given
    # in scrambled (non-sorted) index order -- confirms both that sorting
    # happens correctly AND that exactly the top 3 names are selected, in
    # the correct order.
    X, y = five_feature_dataset
    captured = _spy_selected_features(monkeypatch)
    feature_importance = pd.DataFrame(
        {0: [3, 1, 5, 2, 4]}, index=["f3", "f5", "f1", "f4", "f2"]
    )  # sorted descending by score: f1(5), f2(4), f3(3), f4(2), f5(1)

    partial(X, y, feature_importance, n_features=3, kind="reg", show=False)
    assert captured["features"] == ["f1", "f2", "f3"]


# ---------------------------------------------------------------------------
# variable_importance: one table per method
# ---------------------------------------------------------------------------

LINEAR_CLASSIFIERS = {
    "logistic": LogisticRegression,
    "linear_svc": LinearSVC,
    "ridge": RidgeClassifier,
    "lda": LinearDiscriminantAnalysis,
    "svc_linear": lambda: SVC(kernel="linear"),
    "perceptron": lambda: Perceptron(random_state=0),
    "sgd": lambda: SGDClassifier(random_state=0),
}
NAIVE_BAYES = {"complement": ComplementNB, "multinomial": MultinomialNB, "bernoulli": BernoulliNB}
TREES = {
    "forest": lambda: RandomForestClassifier(n_estimators=30, random_state=0),
    "extra_trees": lambda: ExtraTreesClassifier(n_estimators=30, random_state=0),
    "tree": lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
    "boosting": lambda: GradientBoostingClassifier(random_state=0),
    "adaboost": lambda: AdaBoostClassifier(random_state=0),
}


@pytest.fixture
def bits_xy(rng):
    """Synthetic binary fingerprints: b0 raises the positive class, b1 lowers it, b2-b5 are noise."""
    n, p = 200, 6
    X = pd.DataFrame((rng.random((n, p)) < 0.3).astype(int), columns=[f"b{i}" for i in range(p)])
    y = pd.Series(((2 * X["b0"] - 2 * X["b1"] + rng.normal(scale=0.7, size=n)) > 0).astype(int))
    return X, y


def _switched(X, feature, value):
    switched = X.copy()
    switched[feature] = value
    return switched


@pytest.mark.parametrize(
    "make",
    [
        *LINEAR_CLASSIFIERS.values(),
        *NAIVE_BAYES.values(),
        lambda: MLPClassifier(hidden_layer_sizes=(5,), max_iter=2000, random_state=0),
    ],
    ids=[*LINEAR_CLASSIFIERS, *NAIVE_BAYES, "mlp"],
)
def test_native_signed_importance_recovers_planted_directions(bits_xy, make):
    X, y = bits_xy
    table = variable_importance(make().fit(X, y), list(X.columns))
    assert table.attrs == {"method": "native", "signed": True}
    assert table.loc["b0", "importance"] > 0 > table.loc["b1", "importance"]


@pytest.mark.parametrize("name", LINEAR_CLASSIFIERS)
def test_native_coefficient_is_the_decision_function_step(bits_xy, name):
    # Property: for a linear model, switching one feature from 0 to 1 moves the
    # decision function by exactly that feature's native importance -- which
    # also pins the sign convention (positive pushes towards classes_[1]).
    X, y = bits_xy
    model = LINEAR_CLASSIFIERS[name]().fit(X, y)
    table = variable_importance(model, list(X.columns))
    for feature in X.columns:
        step = model.decision_function(_switched(X, feature, 1)) - model.decision_function(
            _switched(X, feature, 0)
        )
        assert step == pytest.approx(np.full(len(X), table.loc[feature, "importance"]))


@pytest.mark.parametrize("name", NAIVE_BAYES)
def test_native_naive_bayes_log_odds_is_the_joint_log_likelihood_step(bits_xy, name):
    # Property, against scikit-learn's own predict_joint_log_proba: the native
    # importance is the exact change in the class-1-minus-class-0 joint
    # log-likelihood when the feature switches on.
    X, y = bits_xy
    model = NAIVE_BAYES[name]().fit(X, y)
    table = variable_importance(model, list(X.columns))

    def margin(data):
        jll = model.predict_joint_log_proba(data)
        return jll[:, 1] - jll[:, 0]

    for feature in X.columns:
        step = margin(_switched(X, feature, 1)) - margin(_switched(X, feature, 0))
        assert step == pytest.approx(np.full(len(X), table.loc[feature, "importance"]))


def test_native_bernoulli_nb_includes_the_absent_feature_term(bits_xy):
    # BernoulliNB scores absent features too, so log P(x=1|c) differences
    # alone are not its coefficients (the property test above shows the
    # native value is).
    X, y = bits_xy
    model = BernoulliNB().fit(X, y)
    native = variable_importance(model, list(X.columns))["importance"].reindex(X.columns)
    presence_only = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    assert not np.allclose(native.to_numpy(), presence_only)


@pytest.mark.parametrize("name", TREES)
def test_native_tree_importance_is_unsigned_and_finds_planted_features(bits_xy, name):
    X, y = bits_xy
    table = variable_importance(TREES[name]().fit(X, y), list(X.columns))
    assert table.attrs == {"method": "native", "signed": False}
    assert (table["importance"] >= 0).all()
    assert set(table.index[:2]) == {"b0", "b1"}


@pytest.mark.parametrize(
    "make",
    [GaussianNB, KNeighborsClassifier, HistGradientBoostingClassifier, SVC],
    ids=["gaussian_nb", "knn", "hist_gb", "svc_rbf"],
)
def test_native_raises_typeerror_for_models_without_native_importance(bits_xy, make):
    X, y = bits_xy
    with pytest.raises(TypeError, match="method='permutation'"):
        variable_importance(make().fit(X, y), list(X.columns))


@pytest.mark.parametrize(
    "make",
    [
        LogisticRegression,
        ComplementNB,
        lambda: MLPClassifier(hidden_layer_sizes=(3,), max_iter=50, random_state=0),
    ],
    ids=["logistic", "complement_nb", "mlp"],
)
def test_native_rejects_multiclass_models(bits_xy, make):
    X, _ = bits_xy
    three_classes = np.arange(len(X)) % 3
    with pytest.raises(ValueError, match="binary classification"):
        variable_importance(make().fit(X, three_classes), list(X.columns))


@pytest.mark.parametrize(
    "make",
    [LinearRegression, lambda: PLSRegression(n_components=3), lambda: RANSACRegressor(random_state=0)],
    ids=["linear", "pls", "ransac"],
)
def test_native_regression_coefficients_known_answer(rng, make):
    # Known answer: on a noiseless linear target, a linear regressor's native
    # importance is the generating coefficients (RANSAC: its inlier model's).
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=["f1", "f2", "f3"])
    y = 3 * X["f1"] - 2 * X["f3"]
    table = variable_importance(make().fit(X, y), list(X.columns))
    assert table["importance"].reindex(X.columns).to_numpy() == pytest.approx(
        [3.0, 0.0, -2.0], abs=1e-6
    )


def test_importance_table_contract(bits_xy):
    X, y = bits_xy
    table = variable_importance(LogisticRegression().fit(X, y), list(X.columns))
    assert table.index.name == "feature"
    assert sorted(table.index) == sorted(X.columns)
    assert table["importance"].abs().is_monotonic_decreasing


def test_variable_importance_argument_errors(bits_xy):
    X, y = bits_xy
    names = list(X.columns)
    model = LogisticRegression().fit(X, y)
    with pytest.raises(ValueError, match="method must be one of"):
        variable_importance(model, names, "bogus", X=X)
    with pytest.raises(ValueError, match="needs X"):
        variable_importance(model, names, "shap")
    with pytest.raises(ValueError, match="needs y"):
        variable_importance(model, names, "permutation", X=X)
    with pytest.raises(TypeError, match="takes no options"):
        variable_importance(model, names, n_repeats=5)
    with pytest.raises(ValueError, match="feature names"):
        variable_importance(model, names[:-1])


def test_permutation_importance_is_unsigned_with_mean_and_std(bits_xy):
    X, y = bits_xy
    table = variable_importance(
        LogisticRegression().fit(X, y), list(X.columns), "permutation", X=X, y=y, n_repeats=5
    )
    assert table.attrs == {"method": "permutation", "signed": False}
    assert table["importance"].equals(table["mean"])
    assert (table["std"] >= 0).all()
    assert set(table.index[:2]) == {"b0", "b1"}


def test_shap_linear_closed_form_matches_shap_linear_explainer(bits_xy):
    # Known answer from the shap package itself: the closed form is what
    # shap.LinearExplainer computes with an independent masker over the same
    # background.
    shap = pytest.importorskip("shap")
    X, y = bits_xy
    model = LogisticRegression().fit(X, y)
    table = variable_importance(model, list(X.columns), "shap", X=X)
    masker = shap.maskers.Independent(X, max_samples=len(X))
    expected = np.abs(shap.LinearExplainer(model, masker).shap_values(X)).mean(axis=0)
    assert table["mean_abs"].reindex(X.columns).to_numpy() == pytest.approx(expected)


@pytest.mark.parametrize("name", [*LINEAR_CLASSIFIERS, *NAIVE_BAYES])
def test_shap_direction_of_a_linear_model_is_its_coefficient_sign(bits_xy, name):
    X, y = bits_xy
    model = {**LINEAR_CLASSIFIERS, **NAIVE_BAYES}[name]().fit(X, y)
    native = variable_importance(model, list(X.columns))["importance"]
    table = variable_importance(model, list(X.columns), "shap", X=X)
    assert (np.sign(table["importance"]) == np.sign(native.reindex(table.index))).all()


def test_shap_linear_path_needs_no_shap_install(bits_xy, monkeypatch):
    # Linear and Naive Bayes models use the closed form, so they must work
    # where the optional shap extra isn't installed; tree models must say how
    # to install it.
    monkeypatch.setitem(sys.modules, "shap", None)  # makes `import shap` raise ImportError
    X, y = bits_xy
    table = variable_importance(ComplementNB().fit(X, y), list(X.columns), "shap", X=X)
    assert table.loc["b0", "importance"] > 0 > table.loc["b1", "importance"]
    forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    with pytest.raises(ImportError, match=r"mlmolprop\[shap\]"):
        variable_importance(forest, list(X.columns), "shap", X=X)


@pytest.mark.parametrize("name", ["forest", "tree", "boosting"])
def test_shap_tree_explainer_recovers_planted_directions(bits_xy, name):
    # forest/tree give shap one output per class, boosting a single log-odds
    # output -- both layouts must come out as the positive class.
    pytest.importorskip("shap")
    X, y = bits_xy
    table = variable_importance(TREES[name]().fit(X, y), list(X.columns), "shap", X=X)
    assert table.attrs == {"method": "shap", "signed": True}
    assert table.loc["b0", "importance"] > 0 > table.loc["b1", "importance"]


@pytest.mark.parametrize(
    "make", [TREES["adaboost"], KNeighborsClassifier], ids=["adaboost", "knn"]
)
def test_shap_falls_back_to_the_kernel_explainer(bits_xy, make, monkeypatch):
    # adaboost exposes feature_importances_ but TreeExplainer can't read it;
    # knn isn't a tree at all. Both must end up on KernelExplainer.
    shap = pytest.importorskip("shap")
    calls = []
    real = shap.KernelExplainer

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(shap, "KernelExplainer", spy)
    X, y = bits_xy
    table = variable_importance(
        make().fit(X, y), list(X.columns), "shap", X=X.iloc[:5], background=X, n_background=20
    )
    assert calls
    assert table.attrs["signed"] and len(table) == X.shape[1]


def test_positive_output_handles_every_shap_output_layout():
    from mlmolprop.importance import _positive_output

    values = np.arange(6.0).reshape(3, 2)
    assert _positive_output(values) == pytest.approx(values)
    assert _positive_output([-values, values]) == pytest.approx(values)  # shap < 0.45
    assert _positive_output(np.stack([-values, values], axis=2)) == pytest.approx(values)
    with pytest.raises(ValueError, match="binary"):
        _positive_output([values] * 3)
    with pytest.raises(ValueError, match="binary"):
        _positive_output(np.stack([values] * 3, axis=2))


def test_lime_importance_recovers_planted_directions_and_weights_every_feature(bits_xy):
    X, y = bits_xy
    table = variable_importance(
        LogisticRegression().fit(X, y), list(X.columns), "lime", X=X, n_samples=10,
        num_samples=500,
    )
    assert table.attrs == {"method": "lime", "signed": True}
    assert table.loc["b0", "importance"] > 0 > table.loc["b1", "importance"]
    assert (table["n_explained"] == 10).all()


def test_lime_importance_is_reproducible_and_can_be_sparse(bits_xy):
    X, y = bits_xy
    model = LogisticRegression().fit(X, y)
    options = {"X": X, "n_samples": 8, "num_samples": 300, "random_state": 3}
    first = variable_importance(model, list(X.columns), "lime", **options)
    again = variable_importance(model, list(X.columns), "lime", **options)
    pd.testing.assert_frame_equal(first, again)
    sparse = variable_importance(model, list(X.columns), "lime", num_features=2, **options)
    assert (sparse["n_explained"] <= 8).all()
    assert sparse["n_explained"].sum() == 2 * 8  # two features weighted per explained row


def test_flip_importance_of_a_linear_regressor_is_its_coefficients(bits_xy):
    # Known answer: for a model linear in its inputs, switching a bit on moves
    # every prediction by exactly that bit's coefficient.
    X, y = bits_xy
    model = LinearRegression().fit(X, y)
    table = variable_importance(model, list(X.columns), "flip", X=X)
    assert table["importance"].reindex(X.columns).to_numpy() == pytest.approx(model.coef_)


def test_flip_scores_a_classifier_without_predict_proba_by_its_decision_function(bits_xy):
    # Known answer: LinearSVC has no predict_proba, so it is scored by its
    # (linear) decision function, whose step is exactly the coefficient.
    X, y = bits_xy
    model = LinearSVC().fit(X, y)
    table = variable_importance(model, list(X.columns), "flip", X=X)
    assert table["importance"].reindex(X.columns).to_numpy() == pytest.approx(model.coef_[0])


def test_flip_needs_binary_features(regression_xy):
    X, y = regression_xy
    with pytest.raises(ValueError, match="binary"):
        variable_importance(LinearRegression().fit(X, y), list(X.columns), "flip", X=X)


def test_model_agnostic_methods_work_without_feature_names(bits_xy):
    X, y = bits_xy
    model = LogisticRegression().fit(X.to_numpy(), y.to_numpy())
    table = variable_importance(model, list(X.columns), "flip", X=X.to_numpy())
    assert table.loc["b0", "importance"] > 0 > table.loc["b1", "importance"]


def test_model_agnostic_methods_accept_a_plain_object_with_predict_proba(bits_xy):
    # Not a scikit-learn estimator at all (scikit-learn's is_classifier raises
    # on it): having predict_proba is what makes it scored as a classifier.
    X, y = bits_xy
    fitted = LogisticRegression().fit(X.to_numpy(), y.to_numpy())

    class Plain:
        def predict_proba(self, a):
            return fitted.predict_proba(a)

    names = list(X.columns)
    pd.testing.assert_frame_equal(
        variable_importance(Plain(), names, "flip", X=X.to_numpy()),
        variable_importance(fitted, names, "flip", X=X.to_numpy()),
    )


@pytest.mark.slow
def test_model_agnostic_methods_work_on_a_keras_model(bits_xy):
    pytest.importorskip("keras")
    from mlmolprop.model import _build_dl_model

    X, y = bits_xy
    model = _build_dl_model(X.shape[1], (8,), 0.0, "adam", 0.01, True, rs=0)
    model.fit(X.to_numpy(), y.to_numpy(), epochs=5, verbose=0)
    with pytest.raises(TypeError, match="no native importance"):
        variable_importance(model, list(X.columns))
    table = variable_importance(model, list(X.columns), "flip", X=X)
    assert table.attrs["signed"] and len(table) == X.shape[1]


# ---------------------------------------------------------------------------
# Model-free baselines
# ---------------------------------------------------------------------------


def test_enrichment_importance_known_answer():
    X = pd.DataFrame({"a": [1, 1, 0, 0, 0], "b": [0, 1, 1, 1, 0]})
    y = [1, 1, 0, 0, 0]
    table = enrichment_importance(X, y, ["a", "b"])
    assert table.attrs == {"method": "enrichment", "signed": True}
    assert list(table.index) == ["a", "b"]  # |1 - 0| > |1/2 - 2/3|
    a, b = table.loc["a"], table.loc["b"]
    assert (a["count_pos"], a["count_neg"], b["count_pos"], b["count_neg"]) == (2, 0, 1, 2)
    assert a["importance"] == pytest.approx(1.0)
    assert b["importance"] == pytest.approx(0.5 - 2 / 3)
    # pseudocount 0.5: (2 + 0.5) / (2 + 1) over (0 + 0.5) / (3 + 1)
    assert a["log2_ratio"] == pytest.approx(np.log2((2.5 / 3) / (0.5 / 4)))


def test_significance_importance_known_answer():
    # a separates the classes perfectly: 2x2 table [[3, 0], [0, 3]], whose
    # two-sided Fisher p is 2 / C(6, 3) = 0.1; b carries no information
    # ([[2, 1], [1, 2]], p = 1). Benjamini-Hochberg over two tests doubles the
    # smaller p.
    X = pd.DataFrame({"a": [1, 1, 1, 0, 0, 0], "b": [1, 0, 1, 0, 1, 0]})
    y = [1, 1, 1, 0, 0, 0]
    table = significance_importance(X, y, ["a", "b"])
    assert table.loc["a", "p_value"] == pytest.approx(0.1)
    assert table.loc["b", "p_value"] == pytest.approx(1.0)
    assert table.loc["a", "p_adj"] == pytest.approx(0.2)
    assert table.loc["a", "importance"] == pytest.approx(1.0)  # -log10(0.1), enriched in class 1
    assert table.loc["b", "importance"] == pytest.approx(0.0)


def test_significance_sign_follows_the_enriched_class(bits_xy):
    X, y = bits_xy
    table = significance_importance(X, y, list(X.columns))
    assert table.loc["b0", "importance"] > 0 > table.loc["b1", "importance"]
    assert set(table.index[:2]) == {"b0", "b1"}


def test_model_free_tables_need_a_two_class_target(bits_xy):
    X, _ = bits_xy
    with pytest.raises(ValueError, match="two classes"):
        enrichment_importance(X, np.arange(len(X)) % 3, list(X.columns))


# ---------------------------------------------------------------------------
# bootstrap_importance and stable_features
# ---------------------------------------------------------------------------


def test_bootstrap_is_reproducible_and_seed_dependent(bits_xy):
    X, y = bits_xy
    names = list(X.columns)
    first = bootstrap_importance(ComplementNB(), X, y, names, n_boot=10, random_state=1)
    again = bootstrap_importance(ComplementNB(), X, y, names, n_boot=10, random_state=1)
    other = bootstrap_importance(ComplementNB(), X, y, names, n_boot=10, random_state=2)
    pd.testing.assert_frame_equal(first, again)
    assert not np.allclose(first["mean"], other["mean"].reindex(first.index))


def test_bootstrap_resamples_follow_the_documented_sequence(bits_xy):
    # Pins the documented resampling -- RandomState(random_state).choice(n,
    # size=n, replace=True), resample after resample -- so tables computed
    # earlier with the same sequence reproduce exactly.
    X, y = bits_xy
    table = bootstrap_importance(ComplementNB(), X, y, list(X.columns), n_boot=20, random_state=42)
    rng = np.random.RandomState(42)
    records = []
    for _ in range(20):
        idx = rng.choice(len(X), size=len(X), replace=True)
        model = ComplementNB().fit(X.to_numpy()[idx], y.to_numpy()[idx])
        records.append(model.feature_log_prob_[1] - model.feature_log_prob_[0])
    assert table["mean"].reindex(X.columns).to_numpy() == pytest.approx(np.mean(records, axis=0))
    assert table["std"].reindex(X.columns).to_numpy() == pytest.approx(np.std(records, axis=0))


def test_bootstrap_default_refits_a_clone_with_the_same_hyperparameters(bits_xy):
    X, y = bits_xy
    names = list(X.columns)
    cloned = bootstrap_importance(ComplementNB(alpha=3), X, y, names, n_boot=10)
    explicit = bootstrap_importance(
        ComplementNB(), X, y, names, n_boot=10, model_factory=lambda: ComplementNB(alpha=3)
    )
    pd.testing.assert_frame_equal(cloned, explicit)


def test_bootstrap_summary_columns(bits_xy):
    X, y = bits_xy
    table = bootstrap_importance(LogisticRegression(), X, y, list(X.columns), n_boot=15, top_k=2)
    assert table.attrs == {"method": "bootstrap_native", "signed": True}
    assert table["importance"].to_numpy() == pytest.approx(
        (table["mean"] / (table["std"] + 1e-6)).to_numpy()
    )
    assert table["sign_consistency"].between(0, 1).all()
    assert table["top_k_frequency"].sum() == pytest.approx(2)  # each resample picks exactly 2
    assert set(table.index[:2]) == {"b0", "b1"}


def test_bootstrap_of_an_unsigned_method_has_no_sign_consistency(bits_xy):
    X, y = bits_xy
    table = bootstrap_importance(TREES["tree"](), X, y, list(X.columns), n_boot=5)
    assert table.attrs["signed"] is False
    assert "sign_consistency" not in table.columns


def test_bootstrap_balanced_weights_every_fit(bits_xy):
    # Property of "balanced" weights: each class's weights sum to the same total.
    X, y = bits_xy
    seen = []

    class Recording(LogisticRegression):
        def fit(self, X, y, sample_weight=None):
            seen.append((np.asarray(y), sample_weight))
            return super().fit(X, y, sample_weight=sample_weight)

    bootstrap_importance(Recording(), X, y, list(X.columns), n_boot=4, balanced=True)
    assert len(seen) == 4
    for y_fit, weights in seen:
        assert weights[y_fit == 1].sum() == pytest.approx(weights[y_fit == 0].sum())


def test_bootstrap_scores_data_methods_on_out_of_bag_rows(bits_xy, monkeypatch):
    import mlmolprop.importance as importance_module

    X, y = bits_xy
    evaluated = []
    real = importance_module._importance_values

    def spy(model, method, X_eval, y_eval, options):
        evaluated.append(set(X_eval.index))
        return real(model, method, X_eval, y_eval, options)

    monkeypatch.setattr(importance_module, "_importance_values", spy)
    bootstrap_importance(
        LogisticRegression(), X, y, list(X.columns), method="flip", n_boot=3, random_state=7
    )
    rng = np.random.RandomState(7)
    assert len(evaluated) == 3
    for rows in evaluated:
        in_bag = set(rng.choice(len(X), size=len(X), replace=True))
        assert in_bag.isdisjoint(rows)
        assert in_bag | rows == set(range(len(X)))


def _mean_std_table(rows, signed=True):
    """A hand-made table with mean/std columns, like bootstrap_importance's."""
    table = pd.DataFrame(rows, index=["mean", "std"]).T.astype(float)
    table.attrs["signed"] = signed
    return table


def test_stable_features_keeps_both_directions_above_the_threshold():
    table = _mean_std_table({"a": (1.0, 0.1), "b": (-1.0, 0.2), "c": (0.1, 0.2)})
    out = stable_features(table, threshold=2.0)
    assert out["direction"].to_dict() == {"a": 1, "b": -1}
    assert out.loc["b", "snr"] == pytest.approx(1.0 / (0.2 + 1e-6))


def test_stable_features_threshold_is_mean_minus_k_std():
    # |mean| / std >= k is the same cut as |mean| - k * std >= 0.
    table = _mean_std_table({"inside": (2.0, 0.999), "outside": (2.0, 1.001)})
    assert list(stable_features(table, threshold=2.0).index) == ["inside"]


def test_stable_features_drops_features_whose_reference_disagrees():
    table = _mean_std_table({"a": (1.0, 0.1), "b": (-1.0, 0.1)})
    out = stable_features(table, reference=pd.Series({"a": 0.5, "b": 0.5}))
    assert list(out.index) == ["a"]


def test_stable_features_takes_direction_from_reference_for_unsigned_tables():
    table = _mean_std_table({"a": (1.0, 0.1), "b": (0.8, 0.1), "c": (0.9, 0.1)}, signed=False)
    assert (stable_features(table)["direction"] == 0).all()
    out = stable_features(table, reference=pd.Series({"a": 2.0, "b": -1.0, "c": 0.0}))
    assert out["direction"].to_dict() == {"a": 1, "b": -1}  # c has no direction: dropped


def test_stable_features_top_n_caps_each_direction():
    table = _mean_std_table(
        {"p1": (3, 1), "p2": (5, 1), "p3": (4, 1), "n1": (-6, 1), "n2": (-7, 1)}
    )
    out = stable_features(table, threshold=None, top_n=(2, 1))
    assert list(out.index) == ["p2", "p3", "n2"]


def test_stable_features_support_counts_the_favoured_class():
    X = pd.DataFrame({"a": [1, 1, 1, 0, 0], "b": [0, 0, 1, 1, 1]})
    y = [1, 1, 1, 0, 0]
    table = _mean_std_table({"a": (1.0, 0.1), "b": (-1.0, 0.1)})
    # a is carried by 3 positives; b, which favours the negative class, by 2 negatives.
    assert stable_features(table, X=X, y=y)["support"].to_dict() == {"a": 3, "b": 2}
    assert list(stable_features(table, X=X, y=y, min_support=3).index) == ["a"]
    unsigned = _mean_std_table({"a": (1.0, 0.1), "b": (0.5, 0.1)}, signed=False)
    # no direction: support is every compound carrying the feature
    assert stable_features(unsigned, X=X, y=y)["support"].to_dict() == {"a": 3, "b": 3}


def test_a_feature_no_compound_carries_is_stable_but_dropped_by_min_support(rng):
    # A Naive Bayes model gives a feature that never occurs a weight from its
    # smoothing alone, and that weight barely changes between resamples -- so it
    # clears a signal-to-noise threshold, and only a minimum support removes it.
    n = 300
    X = pd.DataFrame((rng.random((n, 5)) < 0.3).astype(int), columns=[f"b{i}" for i in range(5)])
    X["never"] = 0
    y = pd.Series((rng.random(n) < 0.15).astype(int))  # small positive class, no signal
    table = bootstrap_importance(ComplementNB(), X, y, list(X.columns), n_boot=20)
    assert abs(table.loc["never", "importance"]) > 5
    assert "never" in stable_features(table, threshold=2.0).index
    assert "never" not in stable_features(table, threshold=2.0, X=X, y=y, min_support=1).index


def test_stable_features_accepts_a_permutation_table(bits_xy):
    X, y = bits_xy
    names = list(X.columns)
    permutation = variable_importance(
        LogisticRegression().fit(X, y), names, "permutation", X=X, y=y, n_repeats=10
    )
    out = stable_features(permutation, reference=enrichment_importance(X, y, names))
    assert out.loc["b0", "direction"] == 1
    assert out.loc["b1", "direction"] == -1


def test_stable_features_argument_errors():
    table = _mean_std_table({"a": (1.0, 0.1)})
    with pytest.raises(ValueError, match="mean"):
        stable_features(pd.DataFrame({"importance": [1.0]}, index=["a"]))
    with pytest.raises(ValueError, match="needs X and y"):
        stable_features(table, min_support=2)
    with pytest.raises(TypeError, match="DataFrame"):
        stable_features(table, X=np.ones((3, 1)), y=[1, 0, 1])


# ---------------------------------------------------------------------------
# consensus_features
# ---------------------------------------------------------------------------


def test_consensus_known_answer():
    # top_n=2 per side:
    #   m1 up: a, b          down: d
    #   m2 up: a, c          down: b, d
    #   m3 up: b, a          down: e
    tables = {
        "m1": pd.Series({"a": 3.0, "b": 2.0, "c": 1.0, "d": -1.0}),
        "m2": pd.Series({"a": 3.0, "c": 2.0, "b": -2.0, "d": -1.0}),
        "m3": pd.Series({"b": 3.0, "a": 2.0, "e": -1.0}),
    }
    out = consensus_features(tables, top_n=2, min_agree=2)
    assert list(out.index) == ["a", "b", "c", "d", "e"]
    assert out["direction"].to_dict() == {"a": 1, "b": 1, "c": 1, "d": -1, "e": -1}
    assert out["n_agree"].to_dict() == {"a": 3, "b": 2, "c": 1, "d": 2, "e": 1}
    assert out.loc["b", "n_conflict"] == 1
    assert out["selected"].to_dict() == {"a": True, "b": True, "c": False, "d": True, "e": False}
    assert out.loc["a", "mean_rank"] == pytest.approx((1 + 1 + 2) / 3)
    assert np.isnan(out.loc["b", "rank_m2"])  # m2 put b on the other side
    assert out.loc["d", "rank_m2"] == 2


def test_consensus_defaults_to_unanimous():
    tables = {"m1": pd.Series({"a": 1.0, "b": 0.5}), "m2": pd.Series({"a": 1.0, "c": 0.5})}
    out = consensus_features(tables, top_n=1)
    assert out["selected"].to_dict() == {"a": True}
    assert set(consensus_features(tables, top_n=2)["selected"].index) == {"a", "b", "c"}


def test_consensus_gives_unsigned_methods_the_signed_majority_direction():
    unsigned = pd.Series({"a": 0.1, "b": 0.9, "c": 0.5})
    unsigned.attrs["signed"] = False
    tables = {
        "s1": pd.Series({"a": 1.0, "b": -1.0, "c": 0.5}),
        "s2": pd.Series({"a": 2.0, "b": -0.5, "c": 0.1}),
        "u": unsigned,
    }
    out = consensus_features(tables, top_n=1)
    # u's best feature on the positive side (a, c) is c; on the negative side, b.
    assert out.loc["b", "direction"] == -1 and out.loc["b", "selected"]
    assert out.loc["a", "n_agree"] == 2
    assert out.loc["c", "rank_u"] == 1
    # an explicit reference overrides the majority: now u lists b as positive
    flipped = consensus_features(tables, top_n=1, reference=pd.Series({"a": -1, "b": 1, "c": 1}))
    assert flipped.loc["b", "n_conflict"] == 1


def test_consensus_needs_a_direction_for_unsigned_methods():
    unsigned = pd.Series({"a": 1.0})
    unsigned.attrs["signed"] = False
    with pytest.raises(ValueError, match="reference"):
        consensus_features({"u": unsigned})
    out = consensus_features({"u": unsigned}, top_n=1, reference=pd.Series({"a": -2.0}))
    assert out.loc["a", "direction"] == -1


def test_consensus_argument_errors():
    with pytest.raises(ValueError, match="empty"):
        consensus_features({})
    with pytest.raises(ValueError, match="min_agree"):
        consensus_features({"s": pd.Series({"a": 1.0})}, min_agree=2)
    with pytest.raises(ValueError, match="unique"):
        consensus_features({"s": pd.Series([1.0, 2.0], index=["a", "a"])})


def test_consensus_of_real_tables_selects_the_planted_features(bits_xy):
    X, y = bits_xy
    names = list(X.columns)
    tables = {
        "native": variable_importance(LogisticRegression().fit(X, y), names),
        "forest": variable_importance(TREES["forest"]().fit(X, y), names),
        "stability": bootstrap_importance(ComplementNB(), X, y, names, n_boot=20),
        "enrichment": enrichment_importance(X, y, names),
    }
    out = consensus_features(tables, top_n=1)
    assert out[out["selected"]]["direction"].to_dict() == {"b0": 1, "b1": -1}
