"""Tests for mlmolprop.importance."""

import pandas as pd
import pytest

from mlmolprop.importance import lime_explain, partial


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known bug: `selected = top_features[: n_features + 1]` plots "
        "n_features + 1 features, not n_features -- asking for 4 (the "
        "documented default) plots 5."
    ),
)
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


@pytest.mark.xfail(
    strict=True,
    reason="known bug, same as test_partial_plots_exactly_n_features",
)
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
