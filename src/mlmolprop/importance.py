"""Model interpretability: variable importance, its stability, and consensus across methods.

Three layers, each producing the same kind of table so that any of them can be
compared, filtered or combined:

1. **One importance per method.** :func:`variable_importance` is the single
   entry point for every fitted model: ``method="native"`` reads what the
   model exposes about itself (linear coefficients, Naive Bayes log-odds, tree
   importances, MLP connection weights), and the model-agnostic methods
   (``"permutation"``, ``"shap"``, ``"lime"``, ``"flip"``) work for any model
   that predicts. :func:`enrichment_importance` and
   :func:`significance_importance` are model-free baselines from the data alone.
2. **Stability.** :func:`bootstrap_importance` refits the model on bootstrap
   resamples and reports each feature's mean, spread and signal-to-noise ratio;
   :func:`stable_features` keeps the features that clear a signal-to-noise
   threshold, per direction, optionally requiring a minimum number of
   compounds that carry the feature.
3. **Consensus.** :func:`consensus_features` counts how many methods rank each
   feature in their own top ``top_n``, per direction.

The table contract: a DataFrame indexed by feature name (index name
``"feature"``), most important first, whose ``importance`` column is the score
the method ranks features by, plus method-specific columns.
``table.attrs["signed"]`` says whether ``importance`` carries a direction --
positive pushes the prediction up (towards the positive class ``classes_[1]``
for a classifier), negative pushes it down -- or is a magnitude only (tree
importances, permutation). ``table.attrs["method"]`` names the method.

Also here: per-row LIME explanations (:func:`lime_explain`) and partial
dependence plots (:func:`partial`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_VALID_MODES = {"classification", "regression"}
_METHODS = ("native", "permutation", "shap", "lime", "flip")
# Added to a standard deviation before dividing by it, so a feature whose
# importance never varies across resamples gets a large finite signal-to-noise
# ratio instead of a division by zero.
_EPS = 1e-6


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

    selected = top_features[:n_features]
    display = PartialDependenceDisplay.from_estimator(
        model, tr_x, features=selected, feature_names=selected
    )
    plt.tight_layout()
    if show:
        plt.show()
    return display


# ---------------------------------------------------------------------------
# Layer 1: one importance table per method
# ---------------------------------------------------------------------------


def variable_importance(
    model, feature_names, method: str = "native", X=None, y=None, **options
) -> pd.DataFrame:
    """Variable importance of a fitted model, as one table whatever the algorithm.

    Parameters
    ----------
    model : fitted estimator
        Any scikit-learn-compatible model, including the ones returned by
        :func:`mlmolprop.model.Model` and :func:`mlmolprop.model.ModelC`.
        Keras models (``M="dl"``) work with "shap", "lime" and "flip".
    feature_names : list[str]
        Column names, in the order the model was trained on.
    method : {"native", "permutation", "shap", "lime", "flip"}, default "native"
        - ``"native"``: what the fitted model exposes about itself, no data
          needed. Linear coefficients (``coef_``; signed). The per-feature
          log-odds of a ComplementNB, MultinomialNB or BernoulliNB (signed;
          for BernoulliNB this includes the absent-feature term, so it is the
          exact coefficient of the model's linear decision function). Tree
          importances (``feature_importances_``; magnitude only). The
          connection weights of a scikit-learn MLP (the product of its weight
          matrices; it ignores the activations, so it shows direction only, not
          a precise effect). Raises ``TypeError`` for a model that exposes
          none of these (k-nearest neighbours, Gaussian process, GaussianNB,
          HistGradientBoosting, a non-linear SVM, a Keras model); use a
          model-agnostic method for those. Coefficients are comparable across
          features only when the features share a scale (fingerprint bits, or
          standardized descriptors).
        - ``"permutation"``: the drop in score when one feature's column is
          shuffled (:func:`sklearn.inspection.permutation_importance`);
          magnitude only. Needs ``X`` and ``y``, ideally held-out data.
          Options: ``n_repeats=30``, ``random_state=0``, ``scoring=None`` (the
          model's own ``score``: accuracy for a classifier, which barely moves
          on imbalanced data, so consider ``"roc_auc"``), ``n_jobs=None``.
        - ``"shap"``: SHAP values for the rows of ``X``, summarized per feature
          as ``sign(cov(x, shap)) * mean(|shap|)``: the usual mean-|SHAP|
          magnitude, signed by whether larger feature values (a bit being
          present) raise the prediction. The plain mean of SHAP values is not
          used: it is close to zero by construction whenever ``X`` resembles
          the background, and for sparse bits it is dominated by the rows
          where the bit is absent. Linear and Naive Bayes models use the exact
          closed form ``coef * (x - mean(background))``, which needs no shap
          install; tree models use ``shap.TreeExplainer``; anything else uses
          ``shap.KernelExplainer``, which is slow, so keep ``X`` to tens of
          rows. Options: ``background`` (default ``X``),
          ``n_background=100`` (rows sampled from it for the kernel
          explainer), ``random_state=0``. The tree and kernel explainers need
          the optional "shap" extra.
        - ``"lime"``: LIME local linear models fitted around a sample of rows
          of ``X``, averaged per feature (signed). Features are kept
          continuous rather than discretized, so a local weight always means
          "higher value raises the prediction": a discretized weight means
          "being in this row's bin", which flips meaning between rows where a
          bit is present and rows where it is absent. Options:
          ``n_samples=50`` (rows explained), ``num_features=None`` (every
          feature weighted in every row), ``num_samples=5000`` (perturbations
          per row), ``random_state=0``, ``explainer_kwargs`` (passed to
          ``lime.lime_tabular.LimeTabularExplainer``).
        - ``"flip"``: for binary (0/1) features, the mean change in the score
          when the feature is switched from 0 to 1 in every row of ``X``, all
          other features unchanged (signed).

        Every model-agnostic method scores a classifier by its probability of
        the positive class (its decision function if it has no
        ``predict_proba``) and a regressor by its prediction.
    X, y : array-like, optional
        Evaluation data: required by every method except "native" (``y`` only
        by "permutation").
    **options
        Method options, listed under ``method``.

    Returns
    -------
    pandas.DataFrame
        The table described in this module's docstring. Extra columns:
        "permutation" adds ``mean`` and ``std`` over the repeats (so it can go
        straight into :func:`stable_features`); "shap" adds ``mean_abs``;
        "lime" adds ``mean_abs`` and ``n_explained`` (rows in which the
        feature was weighted).
    """
    names = list(feature_names)
    values, signed, extra = _importance_values(model, method, X, y, options)
    return _importance_table(values, names, method, signed, extra)


def _importance_values(model, method, X, y, options):
    """``(values, signed, extra columns)`` for one method, in training-column order."""
    if method == "native":
        if options:
            raise TypeError(f"method='native' takes no options, got {sorted(options)}")
        found = _native_values(model)
        if found is None:
            raise TypeError(
                f"{type(model).__name__} exposes no native importance (no coef_, "
                "feature_log_prob_, feature_importances_ or coefs_); use "
                "method='permutation', 'shap', 'lime' or 'flip' instead"
            )
        return found[0], found[1], {}
    if method not in _METHODS:
        raise ValueError(f"method must be one of {list(_METHODS)}, got {method!r}")
    if X is None:
        raise ValueError(f"method={method!r} needs X")
    if method == "permutation":
        if y is None:
            raise ValueError("method='permutation' needs y")
        return _permutation_values(model, X, y, **options)
    if method == "shap":
        return _shap_values(model, X, **options)
    if method == "lime":
        return _lime_values(model, X, **options)
    return _flip_values(model, X, **options)


def _importance_table(values, names, method, signed, extra=None) -> pd.DataFrame:
    """Wrap one method's per-feature values in the module's table contract."""
    values = np.asarray(values, dtype=float)
    if values.shape != (len(names),):
        raise ValueError(
            f"{method} importance has {values.size} values but {len(names)} feature names "
            "were given"
        )
    table = pd.DataFrame(
        {"importance": values, **(extra or {})}, index=pd.Index(names, name="feature")
    )
    key = np.abs(values) if signed else values
    # argsort of the negated key: most important first, ties kept in training
    # order, NaN last.
    table = table.iloc[np.argsort(-key, kind="stable")]
    table.attrs.update(method=method, signed=bool(signed))
    return table


def _native_values(model):
    """``(values, signed)`` for what a fitted model exposes about itself, or None."""
    coef = _linear_coefficients(model)
    if coef is not None:
        return coef, True
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        return np.asarray(importances, dtype=float), False
    weights = getattr(model, "coefs_", None)
    if weights is not None:
        # "Connection weights" importance (Olden & Jackson, 2002, Ecological
        # Modelling 154:135-150): the product of every weight matrix, input ->
        # ... -> output, sums the weights along every path through the network.
        # Exact only if every activation is the identity; scikit-learn's MLPs
        # default to ReLU, so this is a linear surrogate -- directional, not a
        # precise effect, and rougher the more hidden layers there are.
        product = weights[0] if len(weights) == 1 else np.linalg.multi_dot(weights)
        return _single_output(product.T, model), True
    return None


def _linear_coefficients(model):
    """Coefficients of a model whose decision function is linear in its inputs, or None."""
    from sklearn.linear_model import RANSACRegressor
    from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

    if isinstance(model, RANSACRegressor):
        model = model.estimator_
    if isinstance(model, (BernoulliNB, ComplementNB, MultinomialNB)):
        # The joint log-likelihood of ComplementNB/MultinomialNB is
        # x . feature_log_prob_[c] (+ the class prior), so the difference of the
        # two classes' rows is the coefficient of the class-1-vs-0 margin.
        # (ComplementNB's feature_log_prob_ is already the negated complement.)
        log_prob = model.feature_log_prob_
        if isinstance(model, BernoulliNB):
            # BernoulliNB also scores absent features: its joint log-likelihood
            # is x . (log p - log(1 - p)) + sum(log(1 - p)), so a present
            # feature's coefficient is log p - log(1 - p), not log p alone.
            log_prob = log_prob - np.log1p(-np.exp(log_prob))
        if log_prob.shape[0] != 2:
            raise ValueError(
                f"{type(model).__name__} has {log_prob.shape[0]} classes; variable "
                "importance here covers binary classification and single-target regression"
            )
        return log_prob[1] - log_prob[0]
    coef = getattr(model, "coef_", None)
    if coef is None:
        return None
    coef = coef.toarray() if hasattr(coef, "toarray") else np.asarray(coef, dtype=float)
    return _single_output(np.atleast_2d(coef), model)


def _single_output(matrix, model):
    """The only row of an (outputs, features) matrix; error for multiclass/multi-target."""
    if matrix.shape[0] != 1:
        raise ValueError(
            f"{type(model).__name__} has {matrix.shape[0]} outputs (multiclass or "
            "multi-target); variable importance here covers binary classification and "
            "single-target regression"
        )
    return matrix[0]


def _permutation_values(model, X, y, n_repeats=30, random_state=0, scoring=None, n_jobs=None):
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring,
        n_jobs=n_jobs,
    )
    mean = result.importances_mean
    return mean, False, {"mean": mean, "std": result.importances_std}


def _shap_values(model, X, background=None, n_background=100, random_state=0):
    X_arr = np.asarray(X, dtype=float)
    background = X_arr if background is None else np.asarray(background, dtype=float)
    phi = _shap_matrix(model, X_arr, background, n_background, random_state)
    mean_abs = np.abs(phi).mean(axis=0)
    # Sign of the covariance between each feature and its SHAP values: +1 when
    # larger values (a present bit) go with larger contributions. For a linear
    # model this is exactly the sign of the coefficient.
    centred = (X_arr - X_arr.mean(axis=0)) * (phi - phi.mean(axis=0))
    return np.sign(centred.sum(axis=0)) * mean_abs, True, {"mean_abs": mean_abs}


def _shap_matrix(model, X, background, n_background, random_state):
    """Per-row, per-feature SHAP values for P(positive class) or the prediction."""
    coef = _linear_coefficients(model)
    if coef is not None:
        # Exact interventional SHAP for f(x) = coef . x + b: phi_j = coef_j *
        # (x_j - E[x_j]) -- what shap.LinearExplainer computes with its default
        # independent masker, in the model's margin (log-odds) units.
        return (_linear_inputs(model, X) - _linear_inputs(model, background).mean(axis=0)) * coef
    shap = _import_shap()
    if hasattr(model, "feature_importances_") or type(model).__name__.startswith(
        "HistGradientBoosting"
    ):
        try:
            explainer = shap.TreeExplainer(model)
        except ValueError:
            # shap's InvalidModelError (a ValueError): a tree-like model it
            # doesn't support, e.g. AdaBoost -- fall through to the kernel.
            explainer = None
        if explainer is not None:
            return _positive_output(explainer.shap_values(_model_input(model, X)))
    explainer = shap.KernelExplainer(
        _score_function(model), shap.sample(background, n_background, random_state=random_state)
    )
    return _positive_output(explainer.shap_values(X, silent=True))


def _linear_inputs(model, X):
    """``X`` as the linear decision function sees it (BernoulliNB binarizes first)."""
    from sklearn.naive_bayes import BernoulliNB

    if isinstance(model, BernoulliNB) and model.binarize is not None:
        return (X > model.binarize).astype(float)
    return X


def _positive_output(values):
    """SHAP values of the positive class (or the only output), shape (rows, features)."""
    if isinstance(values, list):  # shap < 0.45 returned one array per class
        if len(values) > 2:
            raise ValueError("variable importance here covers binary classification only")
        values = values[-1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:  # shap >= 0.45: (rows, features, classes)
        if values.shape[2] > 2:
            raise ValueError("variable importance here covers binary classification only")
        values = values[:, :, -1]
    return values


def _import_shap():
    """Import shap, raising an informative error if the optional extra is missing."""
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "method='shap' requires the optional 'shap' extra for non-linear models. "
            "Install with: pip install 'mlmolprop[shap]'"
        ) from e
    return shap


def _lime_values(
    model, X, n_samples=50, num_features=None, num_samples=5000, random_state=0,
    explainer_kwargs=None,
):
    import lime.lime_tabular

    X_arr = np.asarray(X, dtype=float)
    n_rows, n_features = X_arr.shape
    k = n_features if num_features is None else num_features
    rows = np.random.RandomState(random_state).choice(
        n_rows, size=min(n_samples, n_rows), replace=False
    )
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_arr,
        **{
            # Regression mode on the score covers classifiers too: LIME's
            # classification mode regresses the class probability anyway.
            "mode": "regression",
            "discretize_continuous": False,
            "feature_selection": "none" if k >= n_features else "auto",
            "random_state": random_state,
            **(explainer_kwargs or {}),
        },
    )
    score = _score_function(model)
    weights = np.zeros((len(rows), n_features))
    weighted = np.zeros((len(rows), n_features), dtype=bool)
    for r, i in enumerate(rows):
        explanation = explainer.explain_instance(
            X_arr[i], score, num_features=k, num_samples=num_samples
        )
        for j, w in explanation.local_exp[1]:
            weights[r, j] = w
            weighted[r, j] = True
    n_explained = weighted.sum(axis=0)
    denominator = np.maximum(n_explained, 1)
    return (
        weights.sum(axis=0) / denominator,
        True,
        {"mean_abs": np.abs(weights).sum(axis=0) / denominator, "n_explained": n_explained},
    )


def _flip_values(model, X):
    X_arr = np.asarray(X, dtype=float)
    if not np.isin(X_arr, (0.0, 1.0)).all():
        raise ValueError("method='flip' needs binary (0/1) features")
    score = _score_function(model)
    work = X_arr.copy()
    effects = np.empty(X_arr.shape[1])
    for j in range(X_arr.shape[1]):
        work[:, j] = 1.0
        present = score(work)
        work[:, j] = 0.0
        effects[j] = np.mean(present - score(work))
        work[:, j] = X_arr[:, j]
    return effects, True, {}


def _score_function(model):
    """``score(array) -> one value per row``: P(positive class), else the prediction.

    A classifier without ``predict_proba`` is scored by its decision function.
    Arrays are passed to the model as DataFrames when it was fitted on one, so
    scikit-learn doesn't warn about missing feature names.
    """
    from sklearn.base import is_classifier

    if type(model).__module__.split(".")[0] == "keras":

        def raw(a):
            return model.predict(a, verbose=0)

    else:
        try:
            classifier = is_classifier(model)
        except AttributeError:  # not a scikit-learn estimator at all
            classifier = hasattr(model, "predict_proba")
        if classifier and hasattr(model, "predict_proba"):

            def raw(a):
                return model.predict_proba(a)[:, 1]

        elif classifier:
            raw = model.decision_function
        else:
            raw = model.predict

    def score(a):
        out = np.asarray(raw(_model_input(model, a)), dtype=float)
        return out[:, 0] if out.ndim == 2 and out.shape[1] == 1 else out

    return score


def _model_input(model, a):
    """``a`` as a DataFrame with the model's training column names, if it has any."""
    names = getattr(model, "feature_names_in_", None)
    return a if names is None else pd.DataFrame(a, columns=names)


# ---------------------------------------------------------------------------
# Model-free baselines (binary features, binary target)
# ---------------------------------------------------------------------------


def enrichment_importance(X, y, feature_names, pseudocount: float = 0.5) -> pd.DataFrame:
    """How much more often each feature is present in the positive class than the negative.

    A model-free baseline: no fitting, no test, just class-wise presence
    rates. Useful as a sanity check against the model-derived methods, and as
    a ``reference`` direction for methods that report magnitude only.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix; a feature counts as present wherever it is non-zero
        (fingerprint bits or counts).
    y : array-like of two classes
        The positive class is the larger label (``1`` for 0/1 labels), the
        same convention as scikit-learn's ``classes_[1]``.
    feature_names : list[str]
    pseudocount : float, default 0.5
        Added to each count (and twice to each class size) for
        ``log2_ratio``, so a feature absent from one class gets a finite
        ratio.

    Returns
    -------
    pandas.DataFrame
        ``importance`` is the difference in presence rates (positive class
        minus negative class, signed): it favours features carried by many
        compounds, where a ratio would favour rare ones. Also ``rate_pos``,
        ``rate_neg``, ``count_pos``, ``count_neg`` (compounds of each class
        carrying the feature) and ``log2_ratio`` (smoothed ratio of the
        rates).
    """
    present, positive = _presence_by_class(X, y)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    count_pos = present[positive].sum(axis=0)
    count_neg = present[~positive].sum(axis=0)
    rate_pos, rate_neg = count_pos / n_pos, count_neg / n_neg
    c = pseudocount
    log2_ratio = np.log2(((count_pos + c) / (n_pos + 2 * c)) / ((count_neg + c) / (n_neg + 2 * c)))
    return _importance_table(
        rate_pos - rate_neg,
        list(feature_names),
        "enrichment",
        True,
        {
            "rate_pos": rate_pos,
            "rate_neg": rate_neg,
            "count_pos": count_pos,
            "count_neg": count_neg,
            "log2_ratio": log2_ratio,
        },
    )


def significance_importance(X, y, feature_names) -> pd.DataFrame:
    """Per-feature association with a binary target: Fisher's exact test, FDR-corrected.

    For each feature, tests the 2x2 table (present/absent x positive/negative
    class) with a two-sided Fisher's exact test, which stays exact for the
    small counts typical of rare fingerprint bits. P-values are corrected
    across features with Benjamini-Hochberg.

    Parameters
    ----------
    X, y, feature_names
        As for :func:`enrichment_importance`.

    Returns
    -------
    pandas.DataFrame
        ``importance`` is ``-log10(p_value)``, signed by which class the
        feature is enriched in (positive class: +). Also ``p_value``,
        ``p_adj`` (Benjamini-Hochberg), ``count_pos``, ``count_neg``.
    """
    from scipy.stats import false_discovery_control, fisher_exact

    present, positive = _presence_by_class(X, y)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    count_pos = present[positive].sum(axis=0)
    count_neg = present[~positive].sum(axis=0)
    p_value = np.array(
        [
            fisher_exact([[a, n_pos - a], [b, n_neg - b]]).pvalue
            for a, b in zip(count_pos, count_neg, strict=True)
        ]
    )
    direction = np.sign(count_pos / n_pos - count_neg / n_neg)
    return _importance_table(
        direction * -np.log10(np.maximum(p_value, 1e-300)),
        list(feature_names),
        "significance",
        True,
        {
            "p_value": p_value,
            "p_adj": false_discovery_control(p_value, method="bh"),
            "count_pos": count_pos,
            "count_neg": count_neg,
        },
    )


def _presence_by_class(X, y):
    """``(present, positive)``: boolean feature-presence matrix and positive-class row mask."""
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    if len(classes) != 2:
        raise ValueError(f"y must have exactly two classes, got {len(classes)}")
    return np.asarray(X) != 0, y_arr == classes[1]


# ---------------------------------------------------------------------------
# Layer 2: stability
# ---------------------------------------------------------------------------


def bootstrap_importance(
    model,
    X,
    y,
    feature_names,
    method: str = "native",
    n_boot: int = 100,
    random_state: int = 42,
    model_factory=None,
    balanced: bool = False,
    top_k: int | None = None,
    **options,
) -> pd.DataFrame:
    """Stability of a model's variable importance under bootstrap resampling.

    Refits the model on ``n_boot`` bootstrap resamples of ``(X, y)`` and
    records each feature's importance every time. A large mean with a small
    spread is a feature the model relies on whatever the sample; a large
    spread is one whose apparent importance is closer to resampling noise.

    Parameters
    ----------
    model : estimator
        The fitted model to assess; each resample refits an unfitted copy with
        the same hyperparameters (:func:`sklearn.base.clone`).
    X, y : training feature matrix and target.
    feature_names : list[str]
    method : str, default "native"
        Any :func:`variable_importance` method. Methods that need data are
        evaluated on each resample's out-of-bag rows, the rows its fit never
        saw.
    n_boot : int, default 100
    random_state : int, default 42
        Seeds the resampling. Each resample is
        ``RandomState(random_state).choice(n, size=n, replace=True)`` in turn,
        so tables are reproducible run to run.
    model_factory : callable, optional
        ``() -> unfitted estimator``, in place of cloning ``model``, e.g. to
        skip work only the final model needs (``probability=False`` for an
        SVC whose importance is its coefficients).
    balanced : bool, default False
        Pass ``sample_weight=compute_sample_weight("balanced", y_resample)``
        to every fit -- how :func:`mlmolprop.model.ModelC` balances gb and
        xgb, which have no ``class_weight`` argument.
    top_k : int, optional
        Also report how often each feature lands in a resample's top
        ``top_k`` (by absolute importance when signed): the selection
        frequency of stability selection.
    **options
        Passed to :func:`variable_importance` for ``method``.

    Returns
    -------
    pandas.DataFrame
        ``importance`` is the signal-to-noise ratio ``mean / (std + 1e-6)``,
        the stability score features are ranked by (signed when ``method``
        is). Also ``mean`` and ``std`` across resamples; for a signed method
        ``sign_consistency``, the fraction of resamples whose sign matches the
        mean's; and with ``top_k``, ``top_k_frequency``. ``attrs["method"]``
        is ``"bootstrap_<method>"``.
    """
    from sklearn.base import clone
    from sklearn.utils.class_weight import compute_sample_weight

    names = list(feature_names)
    y_arr = np.asarray(y)
    n = len(y_arr)
    factory = model_factory if model_factory is not None else (lambda: clone(model))

    def rows(idx):
        return X.iloc[idx] if isinstance(X, pd.DataFrame) else np.asarray(X)[idx]

    rng = np.random.RandomState(random_state)
    records = np.empty((n_boot, len(names)))
    signed = True
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        fitted = factory()
        fit_kwargs = (
            {"sample_weight": compute_sample_weight("balanced", y_arr[idx])} if balanced else {}
        )
        fitted.fit(rows(idx), y_arr[idx], **fit_kwargs)
        if method == "native":
            X_eval = y_eval = None
        else:
            out_of_bag = np.setdiff1d(np.arange(n), idx)
            X_eval, y_eval = rows(out_of_bag), y_arr[out_of_bag]
        values, signed, _ = _importance_values(fitted, method, X_eval, y_eval, options)
        records[i] = values

    mean = records.mean(axis=0)
    std = records.std(axis=0)
    extra = {"mean": mean, "std": std}
    if signed:
        extra["sign_consistency"] = (np.sign(records) == np.sign(mean)).mean(axis=0)
    if top_k is not None:
        key = np.abs(records) if signed else records
        top = np.argsort(-key, axis=1, kind="stable")[:, :top_k]
        extra["top_k_frequency"] = np.bincount(top.ravel(), minlength=len(names)) / n_boot
    return _importance_table(mean / (std + _EPS), names, f"bootstrap_{method}", signed, extra)


def stable_features(
    table: pd.DataFrame,
    threshold: float | None = 2.0,
    reference=None,
    top_n: int | tuple[int, int] | None = None,
    X=None,
    y=None,
    min_support: int = 0,
) -> pd.DataFrame:
    """Features whose importance clears a signal-to-noise threshold, per direction.

    Parameters
    ----------
    table : pandas.DataFrame
        A table with ``mean`` and ``std`` columns: :func:`bootstrap_importance`
        or ``variable_importance(method="permutation")``.
    threshold : float or None, default 2.0
        Minimum ``|mean| / (std + 1e-6)``: the signal is at least this many
        times its noise, like a t-statistic cutoff (2-3 is a common range).
        Equivalent to requiring ``|mean| - threshold * std > 0``. None keeps
        every feature (e.g. to rank with ``top_n`` alone).
    reference : pandas.Series or DataFrame, optional
        Signed scores per feature (or a table, whose ``importance`` column is
        used), e.g. the fitted model's ``variable_importance``. For a signed
        ``table``, features whose reference sign disagrees with the table's
        are dropped -- the resampled and the fitted model must agree on which
        way a feature pushes. For a magnitude-only ``table`` (tree
        importances), it supplies the direction.
    top_n : int or (int, int), optional
        Keep only the best ``top_n`` per direction by signal-to-noise ratio,
        after every other filter; a tuple sets (positive, negative) sizes
        separately.
    X : pandas.DataFrame, optional
        Feature matrix with the features as columns, for the support count.
    y : array-like, optional
        Two-class target, for the support count.
    min_support : int, default 0
        Minimum number of compounds of the favoured class (the positive class
        for a positive direction, the negative class for a negative one) that
        carry the feature, i.e. have it non-zero. Guards against features
        whose effect rests on one or two compounds. Needs ``X`` and ``y``.
        For fingerprints, use at least 1: a feature no training compound
        carries can still get a weight from the model's smoothing alone (a
        Naive Bayes model gives every unseen feature the same smoothed
        log-odds), and because that weight barely changes between resamples,
        its signal-to-noise ratio comes out high.

    Returns
    -------
    pandas.DataFrame
        The selected features, indexed by name, with ``direction`` (+1, -1,
        or 0 when ``table`` is magnitude-only and no ``reference`` is given),
        ``snr``, ``mean``, ``std`` and, when ``X`` and ``y`` are given,
        ``support``; positive direction first, then by ``snr``.
    """
    missing = {"mean", "std"} - set(table.columns)
    if missing:
        raise ValueError(
            f"table has no {sorted(missing)} column(s); use bootstrap_importance or "
            "variable_importance(method='permutation')"
        )
    mean = table["mean"].astype(float)
    snr = mean.abs() / (table["std"].astype(float) + _EPS)
    signed = table.attrs.get("signed", bool((mean < 0).any()))
    direction = np.sign(mean) if signed else pd.Series(0.0, index=table.index)
    keep = pd.Series(True, index=table.index)
    if reference is not None:
        ref_sign = np.sign(_scores(reference).reindex(table.index)).fillna(0.0)
        if signed:
            keep &= ref_sign == direction
        else:
            direction = ref_sign
            keep &= direction != 0
    if threshold is not None:
        keep &= snr >= threshold

    out = pd.DataFrame(
        {"direction": direction.astype(int), "snr": snr, "mean": mean, "std": table["std"]}
    )
    if X is not None and y is not None:
        out["support"] = _support(X, y, out.index, out["direction"].to_numpy())
        keep &= out["support"] >= min_support
    elif min_support:
        raise ValueError("min_support needs X and y")

    out = out[keep].sort_values(["direction", "snr"], ascending=False, kind="stable")
    if top_n is not None:
        n_pos, n_neg = top_n if isinstance(top_n, tuple) else (top_n, top_n)
        out = pd.concat(
            [
                out[out["direction"] > 0].head(n_pos),
                out[out["direction"] == 0].head(n_pos),
                out[out["direction"] < 0].head(n_neg),
            ]
        )
    out.attrs.update(table.attrs)
    return out


def _support(X, y, features, direction):
    """Compounds of each feature's favoured class that carry it (all compounds if direction 0)."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("the support count needs X as a DataFrame with the features as columns")
    present = X[list(features)].to_numpy() != 0
    if not np.any(direction):
        return present.sum(axis=0)
    _, positive = _presence_by_class(X, y)
    return np.where(
        direction > 0,
        present[positive].sum(axis=0),
        np.where(direction < 0, present[~positive].sum(axis=0), present.sum(axis=0)),
    )


# ---------------------------------------------------------------------------
# Layer 3: consensus
# ---------------------------------------------------------------------------


def consensus_features(
    tables: dict, top_n: int = 35, min_agree: int | None = None, reference=None
) -> pd.DataFrame:
    """Features ranked in the top ``top_n`` by at least ``min_agree`` methods, per direction.

    Each method contributes its own best ``top_n`` features in each
    direction -- separately, so a class whose features score lower than the
    other's still gets its share. A feature is selected when at least
    ``min_agree`` methods list it on the same side.

    Parameters
    ----------
    tables : dict[str, pandas.DataFrame or pandas.Series]
        Method name -> a table from this module (its ``importance`` column is
        used) or a Series of scores. Typical evidence: the native importance,
        ``"shap"``, ``"lime"``, a :func:`bootstrap_importance` table (ranked by
        its signal-to-noise ratio) and :func:`enrichment_importance`.
        A Series without ``attrs["signed"]`` is treated as signed.
    top_n : int, default 35
        How many features each method contributes per direction.
    min_agree : int, optional
        How many methods must list a feature on the same side. Defaults to
        all of them (unanimous), a strict bar that can leave a side empty; a
        majority (``len(tables) // 2 + 1``) is a common looser choice.
    reference : pandas.Series or DataFrame, optional
        Signed scores giving magnitude-only methods (tree importances,
        permutation) a direction. Defaults to the majority sign of the signed
        methods; required when every method is magnitude-only.

    Returns
    -------
    pandas.DataFrame
        One row per feature listed by any method, with ``direction`` (the
        side most methods put it on; 0 on a tie), ``n_agree`` (methods on
        that side), ``n_conflict`` (methods on the other side), ``mean_rank``
        (its average rank, 1 = best, among the methods on its side),
        ``selected`` (``n_agree >= min_agree``), and ``rank_<method>``: its
        rank in each method's list on that side (NaN where the method doesn't
        list it there). Positive direction first, then most agreed, then best
        ranked.

    Notes
    -----
    Each side's ``top_n`` is filled even when a side has fewer genuinely
    important features, so on a small or one-sided feature set noise can make
    the list. Pair the consensus with :func:`stable_features` (or a threshold
    on each table) to keep it out.

    Methods can also disagree systematically. With a small class, rankings by
    model weights (coefficients, Naive Bayes log-odds, their bootstrap mean)
    favour features carried by only a few compounds of that class, while
    SHAP's mean magnitude and enrichment favour features many compounds carry.
    Mixing the two kinds splits the vote, and few features of that class
    reach ``min_agree``.
    """
    if not tables:
        raise ValueError("tables is empty")
    names = list(tables)
    if min_agree is None:
        min_agree = len(names)
    if not 1 <= min_agree <= len(names):
        raise ValueError(f"min_agree must be between 1 and {len(names)}, got {min_agree}")
    scores = {name: _scores(tables[name]) for name in names}
    signed = {name: tables[name].attrs.get("signed", True) for name in names}

    ref_sign = None
    if not all(signed.values()):
        if reference is not None:
            ref_sign = np.sign(_scores(reference))
        elif any(signed.values()):
            ref_sign = np.sign(
                pd.concat([np.sign(scores[n]) for n in names if signed[n]], axis=1).sum(axis=1)
            )
        else:
            raise ValueError(
                "every table is magnitude-only, so none gives a direction; pass reference= "
                "(e.g. an enrichment_importance table)"
            )

    side, rank = {}, {}
    for name in names:
        s = scores[name].dropna()
        if signed[name]:
            up = s[s > 0].sort_values(ascending=False, kind="stable").head(top_n)
            down = s[s < 0].sort_values(kind="stable").head(top_n)
        else:
            rs = ref_sign.reindex(s.index).fillna(0.0)
            up = s[rs > 0].sort_values(ascending=False, kind="stable").head(top_n)
            down = s[rs < 0].sort_values(ascending=False, kind="stable").head(top_n)
        side[name] = pd.concat(
            [pd.Series(1.0, index=up.index), pd.Series(-1.0, index=down.index)]
        )
        rank[name] = pd.concat(
            [
                pd.Series(np.arange(1.0, len(up) + 1), index=up.index),
                pd.Series(np.arange(1.0, len(down) + 1), index=down.index),
            ]
        )
    side = pd.DataFrame(side)
    n_up = (side == 1).sum(axis=1)
    n_down = (side == -1).sum(axis=1)
    direction = np.sign(n_up - n_down).astype(int)
    n_agree = np.maximum(n_up, n_down)
    # Ranks only from the methods that put the feature on its winning side.
    rank = pd.DataFrame(rank)[names].where(side.eq(direction, axis=0))

    out = pd.DataFrame(
        {
            "direction": direction,
            "n_agree": n_agree,
            "n_conflict": np.minimum(n_up, n_down),
            "mean_rank": rank.mean(axis=1),
            "selected": (n_agree >= min_agree) & (direction != 0),
        }
    )
    out = out.join(rank.add_prefix("rank_"))
    out.index.name = "feature"
    return out.sort_values(
        ["direction", "n_agree", "mean_rank"], ascending=[False, False, True], kind="stable"
    )


def _scores(table) -> pd.Series:
    """The score a table ranks by: its ``importance`` column, or the Series itself."""
    scores = table["importance"] if isinstance(table, pd.DataFrame) else table
    if scores.index.has_duplicates:
        raise ValueError("feature names must be unique")
    return scores.astype(float)
