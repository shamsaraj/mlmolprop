"""Tests for chemsar.importance."""

import pandas as pd
import pytest

from chemsar.importance import lime_explain, partial


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
