"""Model interpretability: LIME instance explanations and partial dependence plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_VALID_MODES = {"classification", "regression"}


def lime_explain(
    model,
    x,
    feature_names,
    y,
    mode: str = "regression",
    num_features: int = 8,
    start: int = 0,
    graph: bool = False,
    verbose: bool = False,
) -> list:
    """Explain each row of ``x`` with a LIME tabular explainer.

    Parameters
    ----------
    model : fitted estimator
        Must implement ``predict`` (regression) or ``predict_proba``
        (classification).
    x : pandas.DataFrame
        Feature matrix to explain, row by row.
    feature_names : list[str]
        Column names of ``x``.
    y : array-like
        Observed target values, reported alongside each explanation.
    mode : {"classification", "regression"}, default "regression"
    num_features : int, default 8
        Number of features LIME includes in each local explanation.
    start : int, default 0
        Row index to start from (useful for resuming a long run). Rows
        before ``start`` are left as the placeholder string ``"null"``.
    graph : bool, default False
        If True, also render each explanation via LIME's notebook/pyplot
        display helpers (requires a Jupyter/IPython context).
    verbose : bool, default False
        Print each explanation's details as they're computed.

    Returns
    -------
    list
        One entry per row: ``[index, y[i], prediction, exp.as_list(), exp]``,
        or the string ``"null"`` for rows before ``start``.
    """
    import lime
    import lime.lime_tabular

    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")

    tr = np.array(x)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        tr, feature_names=feature_names, verbose=verbose, mode=mode
    )

    results = len(tr) * ["null"]
    for i in range(start, len(tr)):
        if verbose:
            print("lime is processing molecule", x.index[i])

        if mode == "classification":
            exp = explainer.explain_instance(
                tr[i], model.predict_proba, num_features=num_features
            )
            prediction = exp.predict_proba
        else:
            exp = explainer.explain_instance(
                tr[i], model.predict, num_features=num_features
            )
            prediction = exp.predicted_value

        if verbose:
            print("prediction", prediction)
            print("y[i]", y[i])
            print("exp.score", exp.score)
            print("exp.as_list()", exp.as_list())

        results[i] = [x.index.tolist()[i], y[i], prediction, exp.as_list(), exp]

        if graph:
            exp.show_in_notebook(show_table=True)
            exp.as_pyplot_figure()

    return results


def partial(
    x,
    y,
    feature_importance: pd.DataFrame,
    n_features: int = 4,
    kind: str = "reg",
    show: bool = True,
):
    """Fit a gradient-boosted model and plot partial dependence for its top features.

    Parameters
    ----------
    x : pandas.DataFrame
        Feature matrix.
    y : array-like
        Target values.
    feature_importance : pandas.DataFrame
        Feature ranking with feature names as the index and importance
        scores in column 0, most important first after sorting (e.g. the
        output of a feature-importance ranking step elsewhere in the
        pipeline).
    n_features : int, default 4
        Number of top-ranked features to plot.
    kind : {"reg", "class"}, default "reg"
        Whether to fit a GradientBoostingRegressor or GradientBoostingClassifier.
    show : bool, default True
        Whether to call ``plt.show()`` after plotting.

    Returns
    -------
    sklearn.inspection.PartialDependenceDisplay
    """
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.inspection import PartialDependenceDisplay

    top_features = feature_importance.sort_values(
        by=[0], ascending=False
    ).index.tolist()
    tr_x = np.array(x[top_features])
    tr_y = np.array(y)

    if kind == "reg":
        model = GradientBoostingRegressor(n_estimators=10).fit(tr_x, tr_y)
    elif kind == "class":
        model = GradientBoostingClassifier(n_estimators=10).fit(tr_x, tr_y)
    else:
        raise ValueError(f"kind must be 'reg' or 'class', got {kind!r}")

    selected = top_features[: n_features + 1]
    display = PartialDependenceDisplay.from_estimator(
        model, tr_x, features=selected, feature_names=selected
    )
    plt.tight_layout()
    if show:
        plt.show()
    return display
