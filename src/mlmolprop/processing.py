"""Dataset preprocessing: scaling, feature selection, correlation filtering, splitting."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn import preprocessing
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .correlation import find_correlation
from .descriptors import CI


def _drop_all_zero_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are entirely zero (treats 0 as missing, temporarily)."""
    df = df.replace(0, np.nan)
    df = df.dropna(how="all", axis=1)
    return df.replace(np.nan, 0)


def data_prep(
    datafile=None,
    features=None,
    activities=None,
    Scaled="off",
    Normal="off",
    FS="off",
    Sparse="off",
    cat_columns=None,
    Cor="off",
    ndes=6,
    rs=None,
    newset="off",
    org_v_names=None,
    Xnormalized=None,
    Xscaled=None,
    v_names2=None,
    imputation="off",
    vt=0,
    output="./test.data",
    NAME="name",
    TARGET="IC50",
    REMOVE=None,
    rowremoval=None,
    ratio=0.2,
    selection=None,
    mod="class",
    enet_params=None,
):
    """Load a dataset and run it through the standard QSAR preprocessing pipeline.

    Steps: missing-value imputation, row removal by target value, column
    removal, one-hot encoding of categorical columns, then a train/test
    split, followed by (each individually toggleable) zero-variance
    filtering, normalization, scaling, correlation filtering, and
    univariate feature selection, plus an applicability-domain (CI)
    report. The split happens *before* any of those toggleable steps, and
    each one is fit on the training rows only and applied transform-only
    to the test rows -- so none of *their* statistics (a variance, a
    mean/std, a correlation, an F-test score, ...) is computed from test
    rows. Imputation is the exception: it runs before the split, so its
    column means are taken over the whole input. It is treated as dataset
    curation rather than as part of the fitted pipeline -- and, unlike the
    scalers, the fitted imputer is not returned, so it is not re-applied
    under ``newset="on"``. Pre-impute new compounds yourself if they may
    have missing values. The prepared ``(X, y, ...)`` tuple is also
    pickled to ``output`` for later reuse.

    The input dataset is given one of two ways:

    - ``datafile``: a single CSV already containing NAME + all feature
      columns + TARGET.
    - ``features`` + ``activities``: a precomputed feature matrix (as
      returned by :func:`~mlmolprop.fingerprint.make_fingerprints` or
      :func:`~mlmolprop.descriptors.desc`, or a CSV path to the same
      shape) merged in memory with a small activities table for just
      this call's target/row subset. Lets an expensive feature matrix be
      computed once and reused across many ``data_prep()`` calls (e.g.
      one per prediction target, or one per feature-selection config)
      with no recomputation and no full-matrix file rewrite per call.
      Nothing about ``features`` is chemistry-specific -- any tabular
      feature source works the same way.

    Side effects: writes several diagnostic CSVs to the current working
    directory when the corresponding step runs -- ``cor_mat.csv`` (Cor),
    ``Bonferroni.csv`` (FS="clas"), ``dfout.csv``, and
    ``CI_train.csv``/``CI_test.csv``/``CI_Out_train.csv``/``CI_Out_test.csv``.

    Parameters
    ----------
    datafile : str or None
        Path to the input CSV. Mutually exclusive with ``features``/``activities``.
    features : str, pandas.DataFrame, or None
        Precomputed feature matrix, or a CSV path to one. If a
        ``DataFrame`` lacks a ``NAME`` column, its index is used as the
        name (matching ``make_fingerprints()``'s return shape directly).
        Requires ``activities`` to also be given.
    activities : pandas.DataFrame or None
        Small table with a ``NAME`` column (and a ``TARGET`` column, when
        labels are available -- optional for ``newset="on"`` scoring).
        Requires ``features`` to also be given.
    Scaled, Normal, Sparse : {"on", "off"}
        Apply StandardScaler / Normalizer / MaxAbsScaler to X.
    FS : {"off", "clas", "reg", "rfe_clas", "rfe_reg", "enet_clas", "enet_reg"}
        Feature selection method, keeping (up to) ``ndes`` features:

        - ``"clas"``/``"reg"``: univariate ``SelectKBest`` (F-test) for
          classification/regression.
        - ``"rfe_clas"``/``"rfe_reg"``: recursive feature elimination
          (``RFE``) using a ``LogisticRegression``/``LinearRegression``
          base estimator, eliminating 10% of remaining features per step
          (not the default one-at-a-time) to keep repeated refitting cheap
          on wide feature sets.
        - ``"enet_clas"``/``"enet_reg"``: embedded selection via
          ``SelectFromModel`` wrapping an Elastic-Net-penalized
          ``LogisticRegression``/``ElasticNet``. Pass ``ndes=None`` to
          skip the top-k cap entirely and keep whatever the L1 penalty
          alone doesn't zero out.
    cat_columns : list[str] or None
        Column names to one-hot encode (via ``pandas.get_dummies``)
        before variance filtering. Encoding a new external set
        (``newset="on"``) with categories unseen at training time isn't
        handled -- columns are aligned to ``v_names2`` afterward, so a
        missing dummy column raises a clear KeyError instead of silently
        misaligning.
    Cor : {"on", "off"}
        Drop one of each pair of features whose correlation exceeds 0.9.
    ndes : int or None, default 6
        Number of features kept by ``FS``. ``None`` is only meaningful for
        ``FS="enet_clas"``/``"enet_reg"``, where it means "no fixed count"
        (see ``FS`` above).
    rs : int or None
        Random state for the train/test split.
    newset : {"on", "off"}
        If "on", treat ``datafile`` as a new external set to be prepared
        with already-fitted transformers (``Xnormalized``/``Xscaled``)
        and a previously-recorded feature list (``v_names2``), instead
        of fitting new ones and splitting train/test.
    org_v_names
        Opaque value passed through into the ``newset`` result tuple.
    Xnormalized, Xscaled : fitted sklearn transformer or None
        Previously-fit transformers, reused when ``newset="on"``.
    v_names2 : list[str] or None
        Feature names to select, required when ``newset="on"``.
    imputation : {"on", "off"}
        Mean-impute NaNs (column-wise) before anything else.
    vt : float, default 0
        Variance threshold for dropping near-constant features.
    output : str, default "./test.data"
        Path the result tuple is pickled to.
    NAME : str, default "name"
        Column used as the row index.
    TARGET : str, default "IC50"
        Column used as the prediction target.
    REMOVE : list[str] or None
        Extra columns to drop before splitting X/y.
    rowremoval
        If given, rows where ``df[TARGET] == rowremoval`` are dropped.
    ratio : float, default 0.2
        Test-set fraction for the train/test split. ``ratio=0`` skips the
        split entirely: ``X_train``/``X_test`` (and ``y_train``/``y_test``)
        are both the full dataset, so every downstream step (variance
        filtering, scaling, correlation filtering, feature selection) fits
        on -- and every reported "test" metric evaluates against -- all
        available data, with no held-out set.
    selection : list[str] or None
        If given, restrict to just these feature columns (plus TARGET).
    mod : {"class", "reg"}
        Whether the train/test split is stratified by y (classification)
        or not (regression).
    enet_params : dict or None
        Settings of the elastic net behind ``FS="enet_clas"``/``"enet_reg"``,
        over its defaults (``l1_ratio=0.5``; for classification a ``saga``
        logistic regression with ``max_iter=5000``): e.g. ``C``,
        ``l1_ratio``, ``class_weight``, ``random_state``, ``max_iter``,
        ``tol``. A weaker penalty (larger ``C``) leaves more features
        non-zero, which a large ``ndes`` needs.

    Returns
    -------
    list
        ``newset="on"``: ``[X, y, org_v_names]``.
        Otherwise: ``[X_train, y_train, X_test, y_test, v_names,
        Xnormalized, Xscaled, v_names2]``.
    """
    if datafile is not None:
        df = pd.read_csv(datafile, low_memory=False)
    elif features is not None and activities is not None:
        feat = features if isinstance(features, pd.DataFrame) else pd.read_csv(
            features, low_memory=False
        )
        feat = feat.copy()
        if NAME not in feat.columns:
            # e.g. make_fingerprints()'s natural output: indexed by name,
            # no NAME column yet.
            feat = feat.reset_index()
            feat = feat.rename(columns={feat.columns[0]: NAME})
        feat[NAME] = feat[NAME].astype(str)
        act = activities.copy()
        act[NAME] = act[NAME].astype(str)
        act_cols = [NAME] + ([TARGET] if TARGET in act.columns else [])
        df = pd.merge(act[act_cols], feat, on=NAME, how="inner")
    else:
        raise ValueError(
            "data_prep() requires either `datafile` or both `features` and `activities`"
        )

    if imputation == "on":
        # NOTE: known-fragile path (was left partially broken in the original
        # code: missing_values="NaN" as a string never matches actual NaN
        # floats, so imputation silently did nothing). Fixed to use np.nan.
        from sklearn.impute import SimpleImputer

        df = df.dropna(axis=1, how="all")
        # A row with a missing target has no real label to train/evaluate
        # against -- drop it outright rather than letting the imputer below
        # mean-impute a fake one.
        df = df.dropna(subset=[TARGET])
        target_values = df[TARGET]
        df1 = df.drop([NAME, TARGET], axis=1)
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean", copy=True).fit(
            df1
        )
        df2 = imputer.transform(df1)
        # Columns must come from df1 (the actual remaining columns, by
        # label) -- df.columns.values[1:] assumes NAME is the first column
        # and silently mislabels every column before it otherwise.
        df = pd.DataFrame(df2, columns=df1.columns, index=df.loc[:, NAME])
        df[TARGET] = target_values.values

    if rowremoval is not None:
        df.drop(df.loc[df[TARGET] == rowremoval].index, inplace=True)

    if newset == "on":
        df.columns.values[0] = NAME
        df = df.set_index(NAME)
        X = df
        y = None
        if TARGET in df.columns:
            y = np.ravel(df[[TARGET]])
            X = X.drop(TARGET, axis=1)
            X = _drop_all_zero_columns(X)

        if cat_columns:
            X = pd.get_dummies(X, columns=cat_columns)

        X = X[v_names2]
        if Xnormalized is not None:
            X = Xnormalized.transform(X)
            X = pd.DataFrame(X, columns=v_names2)
        if Xscaled is not None:
            X = Xscaled.transform(X)
            X = pd.DataFrame(X, columns=v_names2)

        result = [X, y, org_v_names]
        return result

    if REMOVE is not None:
        df = df.drop(REMOVE, axis=1)
    if imputation != "on":
        df = df.drop(NAME, axis=1)
    if selection is not None:
        df = df[[TARGET] + list(selection)]

    X = df.drop(TARGET, axis=1)
    X = _drop_all_zero_columns(X)

    if cat_columns:
        X = pd.get_dummies(X, columns=cat_columns)

    index_names = df.index
    X = X.set_index(index_names)

    y = df[[TARGET]]
    y = y.set_index(index_names)

    y_split = np.ravel(y)
    if ratio == 0:
        # No held-out set -- every downstream step below fits on (and every
        # reported "test" metric evaluates against) the entire dataset.
        X_train, X_test, y_train, y_test = X, X, y_split, y_split
    elif mod == "class":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_split, test_size=ratio, random_state=rs, stratify=y_split
        )
    elif mod == "reg":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_split, test_size=ratio, random_state=rs
        )

    # Everything below fits on X_train (and y_train, where relevant) only,
    # then applies transform-only to X_test -- no feature, target, or
    # distributional information from the test set is allowed to leak
    # into what gets fit here (variance/near-constant filtering, scaling,
    # correlation filtering, and univariate feature selection all
    # previously fit on the full pre-split dataset, which biases every
    # downstream R2_test/Q2_CV optimistically since feature selection in
    # particular was seeing the test set's own target values).
    vt_selector = VarianceThreshold(vt).fit(X_train)
    vt_columns = X_train.columns[vt_selector.get_support(indices=True)]
    X_train = pd.DataFrame(
        vt_selector.transform(X_train), columns=vt_columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        vt_selector.transform(X_test), columns=vt_columns, index=X_test.index
    )
    v_names = list(X_train.columns.values)
    v_names2 = v_names

    if Normal == "on":
        Xnormalized = preprocessing.Normalizer().fit(X_train)
        X_train = pd.DataFrame(
            Xnormalized.transform(X_train), columns=v_names, index=X_train.index
        )
        X_test = pd.DataFrame(
            Xnormalized.transform(X_test), columns=v_names, index=X_test.index
        )
    if Scaled == "on":
        Xscaled = preprocessing.StandardScaler().fit(X_train)
        X_train = pd.DataFrame(
            Xscaled.transform(X_train), columns=v_names, index=X_train.index
        )
        X_test = pd.DataFrame(
            Xscaled.transform(X_test), columns=v_names, index=X_test.index
        )
    if Sparse == "on":
        Xscaled = preprocessing.MaxAbsScaler().fit(X_train)
        X_train = pd.DataFrame(
            Xscaled.transform(X_train), columns=v_names, index=X_train.index
        )
        X_test = pd.DataFrame(
            Xscaled.transform(X_test), columns=v_names, index=X_test.index
        )
    if Cor == "on":
        cor_mat = X_train.corr(method="pearson", min_periods=1)
        cor_mat.to_csv("cor_mat.csv")
        col_name = find_correlation(X_train, 0.9)
        X_train = X_train.drop(col_name, axis=1)
        X_test = X_test.drop(col_name, axis=1)
        v_names = list(X_train.columns.values)

    if FS != "off":
        selector = _fit_selector(X_train, y_train, FS, ndes, enet_params)
        if FS == "clas":
            # Must be the feature count SelectKBest actually tested just
            # now (X_train.shape[1]), not v_names2 -- that was fixed back
            # when VarianceThreshold ran, and goes stale if Cor="on" drops
            # correlated features afterward.
            bonferroni = selector.pvalues_ * X_train.shape[1]
            bonferroni_df = pd.DataFrame(bonferroni, index=v_names, columns=["p value"])
            bonferroni_df.to_csv("Bonferroni.csv")
        mask = selector.get_support()
        v_names = X_train.columns[mask]
        X_train = pd.DataFrame(
            selector.transform(X_train), columns=v_names, index=X_train.index
        )
        X_test = pd.DataFrame(
            selector.transform(X_test), columns=v_names, index=X_test.index
        )
        v_names = list(X_train.columns.values)

    y_train_series = pd.Series(y_train, index=X_train.index, name=TARGET)
    y_test_series = pd.Series(y_test, index=X_test.index, name=TARGET)
    dfout = pd.concat(
        [
            pd.concat([y_train_series, X_train], axis=1),
            pd.concat([y_test_series, X_test], axis=1),
        ]
    )
    dfout.to_csv("dfout.csv")

    ci_table = CI(X_train, X_test, v_names, X_train.index, X_test.index, cutoff=3)
    ci_table[0].to_csv("CI_train.csv")
    ci_table[1].to_csv("CI_test.csv")
    ci_table[2].to_csv("CI_Out_train.csv")
    ci_table[3].to_csv("CI_Out_test.csv")

    result = [X_train, y_train, X_test, y_test, v_names, Xnormalized, Xscaled, v_names2]
    object2file(result, output)
    return result


# A coefficient counts as kept by an elastic net when it is not zeroed out. SelectFromModel's
# own default threshold is the mean |coefficient| for any estimator not purely L1-penalised,
# which would keep only the above-average features.
_NONZERO = 1e-5


def _sklearn_older_than(major: int, minor: int) -> bool:
    import re

    import sklearn

    found = re.match(r"(\d+)\.(\d+)", sklearn.__version__)
    return (int(found[1]), int(found[2])) < (major, minor)


def _elastic_net(kind: str, params: dict | None = None):
    """The elastic-net estimator behind ``FS="enet_clas"``/``"enet_reg"``, ``params`` over its defaults."""
    params = dict(params or {})
    if kind == "clas":
        settings = {"solver": "saga", "l1_ratio": 0.5, "max_iter": 5000, **params}
        if "penalty" not in settings and _sklearn_older_than(1, 8):
            # Before scikit-learn 1.8 the penalty defaults to "l2" and l1_ratio is
            # ignored unless it is named; from 1.8 on, l1_ratio sets it.
            settings["penalty"] = "elasticnet"
        return LogisticRegression(**settings)
    return ElasticNet(**{"l1_ratio": 0.5, **params})


def _fit_elastic_net(X, y, kind: str, params: dict | None = None):
    """Fit :func:`_elastic_net` on ``(X, y)``: ``(fitted, converged)``.

    The solver's ConvergenceWarning is caught and returned as ``converged``
    instead of shown, so that a caller fitting many folds can count it; any
    other warning is passed on.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fitted = _elastic_net(kind, params).fit(X, y)
    converged = True
    for w in caught:
        if issubclass(w.category, ConvergenceWarning):
            converged = False
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return fitted, converged


def _filter_columns(X: pd.DataFrame, vt: float = 0.0, cor: float | None = 0.9) -> pd.Index:
    """The columns of ``X`` left by the variance filter and then the correlation filter."""
    columns = X.columns[VarianceThreshold(vt).fit(X).get_support()]
    if cor is not None:
        columns = columns.drop(find_correlation(X[columns], cor))
    return columns


def _cached(memory, func):
    """``func``, cached by ``memory`` (a directory or a joblib.Memory) when one is given."""
    if memory is None:
        return func
    if not hasattr(memory, "cache"):
        from joblib import Memory

        memory = Memory(memory, verbose=0)
    return memory.cache(func)


def _fit_selector(X_train, y_train, FS: str, ndes, enet_params: dict | None = None):
    """Fit :func:`data_prep`'s feature selector ``FS``, keeping (up to) ``ndes`` features."""
    if FS == "clas":
        return SelectKBest(f_classif, k=ndes).fit(X_train, y_train)
    if FS == "reg":
        return SelectKBest(score_func=f_regression, k=ndes).fit(X_train, y_train)
    if FS == "rfe_clas":
        # step=0.1 (drop 10% of remaining features per iteration) instead of
        # RFE's default step=1 -- one-at-a-time elimination means up to
        # ~n_features refits to reach a small ndes, prohibitively slow on
        # wide feature sets (ECFP4 fingerprints are ~2048 columns).
        return RFE(
            LogisticRegression(max_iter=1000), n_features_to_select=ndes, step=0.1
        ).fit(X_train, y_train)
    if FS == "rfe_reg":
        return RFE(LinearRegression(), n_features_to_select=ndes, step=0.1).fit(
            X_train, y_train
        )
    if FS in ("enet_clas", "enet_reg"):
        # Every feature the penalty doesn't zero out, capped at the ndes
        # largest |coefficients| (no cap with ndes=None).
        return SelectFromModel(
            _elastic_net(FS[5:], enet_params), threshold=_NONZERO, max_features=ndes
        ).fit(X_train, y_train)
    raise ValueError(f"Unknown FS: {FS!r}")


def select_features(
    X,
    y,
    FS: str = "off",
    ndes=None,
    vt: float = 0.0,
    cor: float | None = 0.9,
    enet_params: dict | None = None,
    memory=None,
) -> list:
    """Fit the variance filter, the correlation filter and a feature selection; the columns they keep.

    The steps :func:`feature_selection_cv` refits in every fold (and
    :func:`data_prep` runs with ``Cor="on"``), in the same order, fitted on
    the rows given -- for refitting a chosen setting on a training set or on
    every compound.

    Parameters
    ----------
    X : pandas.DataFrame
        Features of the rows to fit on, one column per feature.
    y : array-like
        Their target.
    FS : str, default "off"
        ``"off"`` (only the two filters) or any :func:`data_prep` selector:
        ``"clas"``, ``"reg"``, ``"rfe_clas"``, ``"rfe_reg"``, ``"enet_clas"``,
        ``"enet_reg"``.
    ndes : int or None
        Features kept by ``FS`` (``None`` with an elastic net: every feature
        it doesn't zero out).
    vt, cor : as :func:`feature_selection_cv`.
    enet_params : dict or None
        The elastic net's settings, as :func:`data_prep`'s.
    memory : str, path or joblib.Memory, optional
        Caches the filters and the elastic net, which are slow on wide data:
        a later call with the same rows, target and settings reads them back.

    Returns
    -------
    list
        The kept column names, in ``X``'s order. An elastic net that does not
        converge raises a ConvergenceWarning.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    kept, converged = _select(X, y, FS, ndes, vt, cor, enet_params, memory)
    if converged is False:
        warnings.warn(
            "The elastic net did not converge; raise max_iter (or tol) in enet_params",
            ConvergenceWarning,
            stacklevel=2,
        )
    return kept


def _select(X, y, FS, ndes, vt, cor, enet_params, memory):
    """:func:`select_features`: ``(kept columns, converged)``, ``converged`` None without an elastic net."""
    X = pd.DataFrame(X)
    y = np.asarray(y)
    columns = _cached(memory, _filter_columns)(X, vt, cor)
    X = X[columns]
    if FS == "off":
        return list(columns), None
    if FS in ("enet_clas", "enet_reg"):
        enet, converged = _cached(memory, _fit_elastic_net)(X, y, FS[5:], enet_params)
        mask = SelectFromModel(enet, prefit=True, threshold=_NONZERO, max_features=ndes).get_support()
        return list(X.columns[mask]), converged
    return list(X.columns[_fit_selector(X, y, FS, ndes).get_support()]), None


_CV_FS = ("off", "clas", "rfe_clas", "enet_clas")


def _class_scores(truth, pred, positive) -> dict:
    """:func:`feature_selection_cv`'s classification metrics for one set of predictions."""
    from sklearn.metrics import confusion_matrix, fbeta_score, matthews_corrcoef

    truth, pred = np.asarray(truth), np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(
        truth == positive, pred == positive, labels=[False, True]
    ).ravel()
    return {
        "F2": fbeta_score(truth == positive, pred == positive, beta=2, zero_division=0),
        "MCC": matthews_corrcoef(truth, pred),
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "accuracy": (tp + tn) / len(truth),
    }


def _positive_scores(fitted, X, positive):
    """A fitted classifier's score for the positive class (its probability, else its decision function), or None."""
    if hasattr(fitted, "predict_proba"):
        return fitted.predict_proba(X)[:, list(fitted.classes_).index(positive)]
    if hasattr(fitted, "decision_function"):
        scores = np.asarray(fitted.decision_function(X), dtype=float)
        return scores if fitted.classes_[-1] == positive else -scores
    return None


def _pr_auc(truth, scores, positive) -> float:
    """Area under the precision-recall curve (average precision); NaN without scores or positives."""
    from sklearn.metrics import average_precision_score

    truth = np.asarray(truth) == positive
    if scores is None or not truth.any():
        return np.nan
    return float(average_precision_score(truth, scores))


def _mean(values) -> float:
    """Mean of the values that are not NaN; NaN when none is."""
    values = np.asarray(values, dtype=float)
    return float(values[~np.isnan(values)].mean()) if (~np.isnan(values)).any() else np.nan


def _scores(fitted, X, y, positive) -> dict:
    """Every classification metric of a fitted classifier on ``(X, y)``, PR-AUC included."""
    return {**_class_scores(y, fitted.predict(X), positive),
            "PR_AUC": _pr_auc(y, _positive_scores(fitted, X, positive), positive)}


def feature_selection_cv(
    X,
    y,
    configs,
    model=None,
    vt: float = 0.0,
    cor: float | None = 0.9,
    n_splits: int = 5,
    n_repeats: int = 1,
    rs: int | None = None,
    return_folds: bool = False,
    enet_params: dict | None = None,
    memory=None,
):
    """Cross-validate feature-selection settings, refitting the selection inside every fold.

    Choosing a feature count (or a selection method) by cross-validation is
    only fair when the selection is redone in each fold. Selecting once on
    all the training rows lets every fold's held-out rows help choose the
    features they are then scored on, so the cross-validated score comes out
    optimistic -- more so the fewer features are kept from the more
    candidates. Here, in each fold, the variance filter (``vt``), the
    correlation filter (``cor``) and the selection are fit on the fold's
    training rows only, the same steps in the same order as
    :func:`data_prep` with ``Cor="on"``; a fresh copy of ``model`` is trained
    on the features they keep and predicts the held-out rows.

    Parameters
    ----------
    X : pandas.DataFrame or array-like
        Training features, one column per feature (test rows must not be
        included).
    y : array-like of two classes
        Training classes; the positive class is the larger label (``1`` for
        0/1 labels), as in scikit-learn's ``classes_[1]``.
    configs : iterable of (str, int or None)
        ``(FS, ndes)`` pairs as for :func:`data_prep`: ``FS`` one of
        ``"clas"`` (SelectKBest F-test), ``"rfe_clas"`` (RFE with a logistic
        regression) or ``"enet_clas"`` (elastic-net logistic regression),
        keeping (up to) ``ndes`` features; ``ndes=None`` with ``"enet_clas"``
        keeps every feature the elastic net doesn't zero out. ``("off",
        None)`` is no selection: every feature the two filters leave.
    model : unfitted classifier, or dict of them, optional
        Cloned for every fold and configuration. Defaults to
        ``ComplementNB()``. A dict ``{label: classifier}`` compares several
        models, or one model's hyperparameter settings, on the same folds:
        each fold's filters and selections are fitted once and shared by
        every model, and the output gains a ``model`` column (the label).
    vt : float, default 0.0
        Variance threshold, as ``data_prep``'s ``vt``.
    cor : float or None, default 0.9
        Drop one of each pair of features correlated above this (``None``
        skips the filter).
    n_splits, n_repeats : int
        Stratified k-fold cross-validation, repeated with different
        shuffles.
    rs : int or None
        Seeds the folds.
    return_folds : bool, default False
        Also return every fold's scores, on its held-out rows and on its own
        training rows -- what :func:`best_without_overfitting` needs to
        judge overfitting.
    enet_params : dict or None
        The elastic net's settings, as :func:`data_prep`'s. It is fitted
        once per fold; each ``("enet_clas", ndes)`` keeps the ``ndes``
        largest of its non-zero coefficients, so a large ``ndes`` needs a
        penalty weak enough (large ``C``) to leave that many non-zero.
    memory : str, path or joblib.Memory, optional
        Caches each fold's filters and elastic net: a later call on the same
        rows with the same settings (another model, say) reads them back
        instead of refitting.

    Returns
    -------
    pandas.DataFrame, or (pandas.DataFrame, pandas.DataFrame) with ``return_folds``
        One row per configuration (per model and configuration with a dict
        of models), in the given order: ``model`` (with a dict), ``FS``,
        ``ndes``, ``n_features`` (features kept, averaged over folds), and
        from the pooled out-of-fold predictions of each repeat, averaged over
        the repeats: ``F2`` (recall weighted twice as much as precision),
        ``MCC``, ``recall``, ``specificity``, ``precision`` and
        ``accuracy``; ``PR_AUC``, the area under the precision-recall curve
        of the model's probability (or decision function) for the positive
        class, averaged over the folds (NaN for a model with neither); and,
        when an elastic net is among the configurations, ``unconverged``:
        the folds in which it did not converge (NaN on other rows). With
        ``return_folds``, also one row per configuration and fold:
        ``model`` (with a dict), ``FS``, ``ndes``, ``repeat``, ``fold``
        (numbered across repeats), ``n_features``, the same metrics on the
        fold's held-out rows, and each again on the fold's training rows
        (``F2_train``, ``MCC_train``, ...).
    """
    from sklearn.base import clone
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.naive_bayes import ComplementNB

    configs = [(fs, ndes) for fs, ndes in configs]
    unknown = sorted({fs for fs, _ in configs} - set(_CV_FS))
    if unknown:
        raise ValueError(f"FS must be one of {_CV_FS}, got {unknown}")
    labelled = isinstance(model, dict)
    models = model if labelled else {None: ComplementNB() if model is None else model}
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    positive = np.unique(y)[-1]
    filter_columns = _cached(memory, _filter_columns)
    fit_enet = _cached(memory, _fit_elastic_net)

    folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=rs)
    keys = [(label, fs, ndes) for label in models for fs, ndes in configs]
    predicted = {k: np.empty((n_repeats, len(y)), dtype=y.dtype) for k in keys}
    pr_auc = {k: [] for k in keys}
    kept = {c: [] for c in configs}
    unconverged = 0
    fold_rows = []
    for i, (train, held_out) in enumerate(folds.split(X, y)):
        X_train, X_held, y_train = X.iloc[train], X.iloc[held_out], y[train]
        columns = filter_columns(X_train, vt, cor)
        X_train, X_held = X_train[columns], X_held[columns]
        enet = None  # one elastic-net fit per fold serves every enet setting
        for fs, ndes in configs:
            if fs == "off":
                mask = np.ones(len(columns), dtype=bool)
            elif fs == "enet_clas":
                if enet is None:
                    enet, converged = fit_enet(X_train, y_train, "clas", enet_params)
                    unconverged += not converged
                mask = SelectFromModel(
                    enet, prefit=True, threshold=_NONZERO, max_features=ndes
                ).get_support()
            else:
                mask = _fit_selector(X_train, y_train, fs, ndes).get_support()
            kept[fs, ndes].append(int(mask.sum()))
            for label, estimator in models.items():
                fitted = clone(estimator).fit(X_train.loc[:, mask], y_train)
                held = _scores(fitted, X_held.loc[:, mask], y[held_out], positive)
                predicted[label, fs, ndes][i // n_splits, held_out] = fitted.predict(X_held.loc[:, mask])
                pr_auc[label, fs, ndes].append(held["PR_AUC"])
                if return_folds:
                    train_scores = _scores(fitted, X_train.loc[:, mask], y_train, positive)
                    fold_rows.append({
                        **({"model": label} if labelled else {}),
                        "FS": fs, "ndes": ndes, "repeat": i // n_splits, "fold": i,
                        "n_features": int(mask.sum()),
                        **held,
                        **{f"{k}_train": v for k, v in train_scores.items()},
                    })

    has_enet = any(fs == "enet_clas" for fs, _ in configs)
    rows = []
    for label, fs, ndes in keys:
        per_repeat = pd.DataFrame([_class_scores(y, p, positive) for p in predicted[label, fs, ndes]])
        rows.append({**({"model": label} if labelled else {}),
                     "FS": fs, "ndes": ndes, "n_features": float(np.mean(kept[fs, ndes])),
                     **per_repeat.mean().to_dict(), "PR_AUC": _mean(pr_auc[label, fs, ndes]),
                     **({"unconverged": unconverged if fs == "enet_clas" else np.nan} if has_enet else {})})
    table = pd.DataFrame(rows)
    return (table, pd.DataFrame(fold_rows)) if return_folds else table


def resplit_evaluation(
    X,
    y,
    model,
    FS: str = "off",
    ndes=None,
    n_resplits: int = 100,
    test_size: float = 0.2,
    vt: float = 0.0,
    cor: float | None = 0.9,
    enet_params: dict | None = None,
    rs: int | None = None,
    shuffle_labels: bool = False,
    importance: bool = True,
    memory=None,
):
    """Score one recipe -- the filters, a feature selection and a classifier -- on repeated random splits.

    One held-out test set with few positives measures a classifier only
    roughly, and a second recipe on the same split cannot be told apart from
    it unless they differ a lot. Here the whole recipe is refitted on each
    of ``n_resplits`` stratified random training sets (the filters and the
    selection too, as :func:`select_features`) and scored on the matching
    test set. The same ``X``, ``y`` and ``rs`` give the same resplits, so
    two recipes can be compared resplit by resplit with
    :func:`corrected_resampled_ttest` (``n_splits = round(1 / test_size)``
    gives its n_test / n_train).

    Parameters
    ----------
    X : pandas.DataFrame
        Features of every row, one column per feature.
    y : array-like of two classes
        Their classes; the positive class is the larger label.
    model : unfitted classifier
        Cloned for every resplit.
    FS, ndes, vt, cor, enet_params, memory
        The selection, as :func:`select_features`. ``memory`` lets
        recipes that share the resplits and the selection settings (other
        classifiers, or other ``ndes`` with an elastic net) reuse its fits.
    n_resplits : int, default 100
    test_size : float, default 0.2
    rs : int or None
        Seeds the resplits (and the label shuffles).
    shuffle_labels : bool, default False
        Train on randomly permuted training labels and score on the true
        test labels: the chance-level baseline of y-randomization.
    importance : bool, default True
        Also return each resplit's native importance of the features it
        kept (:func:`mlmolprop.importance.variable_importance`).

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame or None)
        One row per resplit: ``resplit``, ``n_features``, the metrics of
        :func:`feature_selection_cv` on its test rows (``F2``, ``MCC``,
        ``recall``, ``specificity``, ``precision``, ``accuracy``,
        ``PR_AUC``) and on its training rows (``F2_train``, ...), and, for
        an elastic net, ``converged``. Then, with ``importance``, one row per
        resplit and one column per feature of ``X``: the importance of each
        feature that resplit kept, NaN for the others (else None).
    """
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedShuffleSplit

    from .importance import variable_importance

    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    positive = np.unique(y)[-1]
    shuffler = np.random.default_rng(rs)
    splits = StratifiedShuffleSplit(n_splits=n_resplits, test_size=test_size, random_state=rs)
    rows, importances = [], []
    for i, (train, test) in enumerate(splits.split(X, y)):
        y_train = shuffler.permutation(y[train]) if shuffle_labels else y[train]
        kept, converged = _select(X.iloc[train], y_train, FS, ndes, vt, cor, enet_params, memory)
        fitted = clone(model).fit(X.iloc[train][kept], y_train)
        rows.append({
            "resplit": i, "n_features": len(kept),
            **_scores(fitted, X.iloc[test][kept], y[test], positive),
            **{f"{k}_train": v for k, v in _scores(fitted, X.iloc[train][kept], y_train, positive).items()},
            **({"converged": converged} if converged is not None else {}),
        })
        if importance:
            values = variable_importance(fitted, kept)["importance"]
            importances.append(values.reindex(X.columns).rename(i))
    scores = pd.DataFrame(rows)
    if not importance:
        return scores, None
    table = pd.DataFrame(importances)
    table.index.name = "resplit"
    return scores, table


def corrected_resampled_ttest(a, b=None, n_splits: int = 5, alternative: str = "two-sided"):
    """Paired t-test across the folds of repeated k-fold cross-validation, corrected for their overlap.

    Folds of a cross-validation share most of their training rows, so their
    scores are correlated, and a plain paired t-test across them is badly
    overconfident. The corrected resampled t-test (Nadeau and Bengio, 2003;
    for repeated k-fold cross-validation, Bouckaert and Frank, 2004) inflates
    the variance of the mean difference from ``s**2 / J`` to
    ``(1 / J + n_test / n_train) * s**2``, with ``J`` the number of folds and
    ``n_test / n_train = 1 / (n_splits - 1)``.

    Parameters
    ----------
    a, b : array-like
        Two models' scores on the same folds, in the same order; with
        ``b=None``, ``a`` holds the per-fold differences.
    n_splits : int, default 5
        Folds per repeat (the k of k-fold cross-validation).
    alternative : {"two-sided", "greater", "less"}
        The alternative hypothesis for the mean of ``a - b``.

    Returns
    -------
    (float, float)
        The t statistic and its p-value, on ``J - 1`` degrees of freedom.
        Identical scores give ``t = 0``.
    """
    from scipy import stats

    d = np.asarray(a, dtype=float) - (0.0 if b is None else np.asarray(b, dtype=float))
    if d.size < 2:
        raise ValueError("Need the scores of at least two folds")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"alternative must be 'two-sided', 'greater' or 'less', got {alternative!r}")
    var = d.var(ddof=1)
    if var == 0:
        t = 0.0 if d.mean() == 0 else float(np.copysign(np.inf, d.mean()))
    else:
        t = float(d.mean() / np.sqrt((1 / d.size + 1 / (n_splits - 1)) * var))
    dist = stats.t(d.size - 1)
    p = {"two-sided": 2 * dist.sf(abs(t)), "greater": dist.sf(t), "less": dist.cdf(t)}[alternative]
    return t, float(min(p, 1.0))


def paired_comparison(
    scores, reference=None, n_splits: int = 5, alpha: float = 0.05, margin: float | None = None
) -> pd.DataFrame:
    """Compare models scored on the same folds or resplits, each against one reference.

    Every difference is tested with :func:`corrected_resampled_ttest`, and
    its confidence interval is widened the same way, so overlapping training
    sets are not counted as independent evidence.

    Parameters
    ----------
    scores : pandas.DataFrame
        One row per fold or resplit, one column per model, the same splits
        in the same order for every model.
    reference : str, optional
        The column every model is compared with. Defaults to the one with
        the highest mean.
    n_splits : int, default 5
        As :func:`corrected_resampled_ttest`: 5 for 5-fold cross-validation,
        or for 80:20 resplits.
    alpha : float, default 0.05
        Level of the two-sided test; the interval of the difference has
        confidence 1 - alpha.
    margin : float, optional
        An equivalence margin. With it, a difference whose (1 - 2 alpha)
        interval lies within +-margin is equivalent (two one-sided tests),
        and each model gets a verdict.

    Returns
    -------
    pandas.DataFrame
        One row per model, highest mean first: ``mean``, ``std``,
        ``difference`` (model minus reference, averaged over the splits),
        ``low`` and ``high`` (its corrected interval) and ``p`` (two-sided;
        NaN for the reference). With ``margin``, also ``equivalent`` and
        ``verdict``: "reference", "better" or "worse" (significant),
        "better (equiv.)" or "worse (equiv.)" (significant, but within the
        margin), "equivalent" (not significant, within the margin) or
        "inconclusive" (neither).
    """
    from scipy import stats

    scores = pd.DataFrame(scores).astype(float)
    reference = scores.mean().idxmax() if reference is None else reference
    n = len(scores)
    if n < 2:
        raise ValueError("Need the scores of at least two splits")
    inflate = 1 / n + 1 / (n_splits - 1)
    rows = []
    for model in scores.columns:
        d = scores[model] - scores[reference]
        se = float(np.sqrt(inflate * d.var(ddof=1)))
        half = stats.t(n - 1).ppf(1 - alpha / 2) * se
        row = {"model": model, "mean": scores[model].mean(), "std": scores[model].std(ddof=1),
               "difference": d.mean(), "low": d.mean() - half, "high": d.mean() + half,
               "p": np.nan if model == reference else corrected_resampled_ttest(d, n_splits=n_splits)[1]}
        if margin is not None:
            narrow = stats.t(n - 1).ppf(1 - alpha) * se
            equivalent = bool(-margin < d.mean() - narrow and d.mean() + narrow < margin)
            significant = model != reference and row["p"] < alpha
            if model == reference:
                verdict = "reference"
            elif significant:
                verdict = ("better" if d.mean() > 0 else "worse") + (" (equiv.)" if equivalent else "")
            else:
                verdict = "equivalent" if equivalent else "inconclusive"
            row.update(equivalent=equivalent, verdict=verdict)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    return table.sort_values("mean", ascending=False, kind="stable")


def holdout_scores(
    truth, predicted, scores=None, n_boot: int = 2000, confidence: float = 0.95, rs: int | None = None
) -> pd.DataFrame:
    """A classifier's metrics on one held-out set, with confidence intervals.

    A test set with few positives measures recall and precision roughly; the
    intervals say how roughly. Recall, specificity, precision and accuracy
    are proportions and get the exact (Clopper-Pearson) binomial interval.
    F2, MCC and PR-AUC get the percentile interval of a stratified bootstrap:
    positives and negatives are resampled separately, so every resample keeps
    the test set's class counts.

    Parameters
    ----------
    truth, predicted : array-like
        True and predicted classes; the positive class is the larger label of
        ``truth``.
    scores : array-like, optional
        The model's score for the positive class (probability or decision
        function), for PR-AUC.
    n_boot : int, default 2000
    confidence : float, default 0.95
    rs : int or None
        Seeds the bootstrap.

    Returns
    -------
    pandas.DataFrame
        One row per metric (``F2``, ``MCC``, ``recall``, ``specificity``,
        ``precision``, ``accuracy``, and ``PR_AUC`` with ``scores``):
        ``value``, ``low``, ``high``, ``interval`` ("exact" or "bootstrap")
        and ``count`` (e.g. "12/15" for a proportion).
    """
    from scipy.stats import beta

    truth, predicted = np.asarray(truth), np.asarray(predicted)
    positive = np.unique(truth)[-1]
    actual, called = truth == positive, predicted == positive
    tail = (1 - confidence) / 2

    def exact(k, n):
        low = beta.ppf(tail, k, n - k + 1) if k > 0 else 0.0
        high = beta.ppf(1 - tail, k + 1, n - k) if k < n else 1.0
        return float(low), float(high)

    def f2_mcc(a, c):
        tp, fn = np.sum(a & c), np.sum(a & ~c)
        fp, tn = np.sum(~a & c), np.sum(~a & ~c)
        f2 = 5 * tp / (5 * tp + 4 * fn + fp) if tp + fn + fp else 0.0
        denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return f2, (tp * tn - fp * fn) / denominator if denominator else 0.0

    tp, fn = int(np.sum(actual & called)), int(np.sum(actual & ~called))
    fp, tn = int(np.sum(~actual & called)), int(np.sum(~actual & ~called))
    proportions = {"recall": (tp, tp + fn), "specificity": (tn, tn + fp),
                   "precision": (tp, tp + fp), "accuracy": (tp + tn, len(truth))}

    rng = np.random.default_rng(rs)
    pos, neg = np.flatnonzero(actual), np.flatnonzero(~actual)
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        row = dict(zip(("F2", "MCC"), f2_mcc(actual[idx], called[idx]), strict=True))
        if scores is not None:
            row["PR_AUC"] = _pr_auc(actual[idx], np.asarray(scores)[idx], True)
        boot.append(row)
    boot = pd.DataFrame(boot)

    values = dict(zip(("F2", "MCC"), f2_mcc(actual, called), strict=True))
    if scores is not None:
        values["PR_AUC"] = _pr_auc(actual, scores, True)
    rows = []
    for metric in ("F2", "MCC"):
        rows.append({"metric": metric, "value": values[metric], "low": boot[metric].quantile(tail),
                     "high": boot[metric].quantile(1 - tail), "interval": "bootstrap", "count": ""})
    for metric, (k, n) in proportions.items():
        rows.append({"metric": metric, "value": k / n if n else 0.0, **dict(zip(("low", "high"), exact(k, n),
                     strict=True)), "interval": "exact", "count": f"{k}/{n}"})
    if scores is not None:
        rows.append({"metric": "PR_AUC", "value": values["PR_AUC"], "low": boot["PR_AUC"].quantile(tail),
                     "high": boot["PR_AUC"].quantile(1 - tail), "interval": "bootstrap", "count": ""})
    return pd.DataFrame(rows).set_index("metric")


def _setting_key(frame: pd.DataFrame) -> pd.Series:
    """One label per (model, FS, ndes), with ndes=None/NaN as "all"; no model column is one model."""
    ndes = frame["ndes"].map(lambda n: "all" if pd.isna(n) else str(int(n)))
    key = frame["FS"].astype(str) + " " + ndes
    return frame["model"].astype(str) + " | " + key if "model" in frame.columns else key


def best_without_overfitting(
    table: pd.DataFrame,
    folds: pd.DataFrame,
    metric: str = "MCC",
    alpha: float = 0.05,
    n_splits: int | None = None,
    max_gap: float | None = None,
) -> pd.DataFrame:
    """Choose a feature-selection setting: the best cross-validated score that does not overfit.

    Takes :func:`feature_selection_cv`'s ``table`` and ``folds``
    (``return_folds=True``); with a dict of models, every model and setting
    is a candidate of its own. A setting's overfitting is its gap: ``metric``
    on each fold's own training rows minus on its held-out rows, averaged
    over the folds. Every model fits its training rows better than unseen
    ones, so the gaps are compared, not tested against zero:

    0. with ``max_gap``, only the settings whose gap is at most that share of
       their training score are candidates (a limit on overfitting, set in
       advance; the steps below then run among them);
    1. competitive: the settings whose held-out ``metric`` is not
       significantly below the best setting's (one-sided
       :func:`corrected_resampled_ttest` on the per-fold scores, ``alpha``);
    2. the reference gap: the smallest gap among the competitive settings;
    3. overfitting: a gap significantly larger than the reference's
       (one-sided test on the per-fold gaps, ``alpha``);
    4. chosen: the competitive setting with the highest cross-validated
       ``metric`` that does not overfit (ties: fewer features).

    There is no multiple-testing correction: passing over a setting that
    overfits is the costlier mistake, and whichever competitive setting is
    chosen instead scores about as well. The steps compare settings with one
    another, so on their own they still choose one when every setting
    overfits; ``max_gap`` rules that out.

    Parameters
    ----------
    table, folds : pandas.DataFrame
        :func:`feature_selection_cv` with ``return_folds=True``.
    metric : str, default "MCC"
        Any of its metrics: ``"MCC"``, ``"F2"``, ``"recall"``, ...
    alpha : float, default 0.05
        Significance level of both one-sided tests.
    n_splits : int, optional
        For :func:`corrected_resampled_ttest`; defaults to the folds per
        repeat in ``folds``. Pass 5 for scores from 80:20 resplits (e.g.
        :func:`resplit_evaluation`'s, with each resplit as a ``fold`` and
        ``repeat`` set to 0).
    max_gap : float, optional
        The largest gap allowed, as a share of the training score: with 0.2,
        a setting whose held-out ``metric`` is more than 20% below its
        training ``metric`` is never chosen.

    Returns
    -------
    pandas.DataFrame
        ``table`` sorted by ``metric`` (best first, then fewer features),
        with ``<metric>_train`` (mean over the folds), ``gap``,
        ``p_below_best`` and ``competitive`` (step 1; ``p_below_best`` is
        NaN for the best itself), ``p_overfits`` and ``overfits`` (step 3;
        NaN and False for the reference itself) and ``chosen``, True on
        exactly one row. With ``max_gap``, also ``gap_share`` (the gap over
        the training score) and ``within_max_gap``; a setting beyond it is
        never competitive, and when no setting is within it ``chosen`` is
        False on every row.
    """
    folds = folds.assign(setting=_setting_key(folds), gap=folds[f"{metric}_train"] - folds[metric])
    if n_splits is None:
        n_splits = int(folds.groupby("repeat")["fold"].nunique().max())
    held = folds.pivot(index="fold", columns="setting", values=metric)
    gaps = folds.pivot(index="fold", columns="setting", values="gap")

    out = table.assign(setting=_setting_key(table))
    out[f"{metric}_train"] = out["setting"].map(folds.groupby("setting")[f"{metric}_train"].mean())
    out["gap"] = out["setting"].map(gaps.mean())
    out = out.sort_values([metric, "n_features"], ascending=[False, True], kind="stable")

    eligible = pd.Series(True, index=out.index)
    if max_gap is not None:
        train = out[f"{metric}_train"]
        out["gap_share"] = (out["gap"] / train).where(train > 0)
        out["within_max_gap"] = out["gap_share"] <= max_gap
        eligible = out["within_max_gap"]
        if not eligible.any():
            return out.drop(columns="setting").assign(
                p_below_best=np.nan, competitive=False, p_overfits=np.nan, overfits=False, chosen=False
            )

    best = out.loc[eligible, "setting"].iloc[0]
    out["p_below_best"] = [
        np.nan if s == best else corrected_resampled_ttest(held[best], held[s], n_splits, "greater")[1]
        for s in out["setting"]
    ]
    out["competitive"] = eligible & ((out["setting"] == best) | (out["p_below_best"] >= alpha))
    reference = out.loc[out["competitive"], "gap"].idxmin()
    ref = out.loc[reference, "setting"]
    out["p_overfits"] = [
        np.nan if s == ref else corrected_resampled_ttest(gaps[s], gaps[ref], n_splits, "greater")[1]
        for s in out["setting"]
    ]
    out["overfits"] = out["p_overfits"] < alpha
    out["chosen"] = False
    out.loc[out.index[out["competitive"] & ~out["overfits"]][0], "chosen"] = True
    return out.drop(columns="setting")


def scaffold_split(names, smiles, test_size: float = 0.2, rs: int | None = None):
    """Train/test split grouped by Bemis-Murcko scaffold, not by row.

    A plain ``train_test_split()`` can put near-identical structural analogs of the same
    compound on both sides of the split -- inflating apparent generalization on datasets
    that include such analogs (e.g. an "analog expansion" test-set design), since the
    model effectively gets to see a close relative of the test compound during training.
    Grouping by scaffold before splitting keeps every compound that shares a scaffold on
    the same side.

    Parameters
    ----------
    names : sequence
        Row identifiers (e.g. compound names), aligned with ``smiles``.
    smiles : sequence
        SMILES strings, one per entry in ``names``.
    test_size : float, default 0.2
        Target fraction of *compounds* (not scaffold groups) in the test split --
        ``GroupShuffleSplit`` targets this at the group level, so the realized fraction
        can differ somewhat from ``test_size`` when scaffold group sizes are uneven.
    rs : int or None
        Random state for the group-assignment shuffle.

    Returns
    -------
    tuple
        ``(train_names, test_names)`` -- ``names`` entries assigned to each side.
    """
    names = list(names)
    scaffolds = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Unparseable SMILES can't be scaffold-matched against anything else --
            # fall back to a group of its own (the raw SMILES string) rather than
            # raising, so one bad structure doesn't block the whole split.
            scaffolds.append(smi)
            continue
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
    train_idx, test_idx = next(gss.split(names, groups=scaffolds))
    return [names[i] for i in train_idx], [names[i] for i in test_idx]


def VarianceThreshold_selector(
    data: pd.DataFrame, threshold2: float = 0
) -> pd.DataFrame:
    """Drop features with variance <= ``threshold2``."""
    columns = data.columns
    selector = VarianceThreshold(threshold2).fit(data)
    labels = [columns[i] for i in selector.get_support(indices=True)]
    return pd.DataFrame(selector.transform(data), columns=labels)


def SelectKBest_selector(
    data: pd.DataFrame, target, ndes: int, kind: str
) -> pd.DataFrame:
    """Select the top ``ndes`` features by univariate F-test score."""
    columns = data.columns
    if kind == "clas":
        selector = SelectKBest(f_classif, k=ndes).fit(data, target)
    elif kind == "reg":
        selector = SelectKBest(f_regression, k=ndes).fit(data, target)
    else:
        raise ValueError(f"kind must be 'clas' or 'reg', got {kind!r}")
    labels = [columns[i] for i in selector.get_support(indices=True)]
    return pd.DataFrame(selector.transform(data), columns=labels)


def list2file(list1, file1) -> None:
    with open(file1, "w") as filehandle:
        filehandle.writelines(f"{place}\n" for place in list1)


def file2list(file1) -> list:
    with open(file1, "r") as filehandle:
        return [line.rstrip() for line in filehandle]


def object2file(object1, file1) -> None:
    with open(file1, "wb") as filehandle:
        pickle.dump(object1, filehandle)


def file2object(file1):
    with open(file1, "rb") as filehandle:
        return pickle.load(filehandle)


def average_bygroup(df: pd.DataFrame, groups, out: str) -> pd.DataFrame:
    """Group ``df`` by ``groups``, average, transpose, and write to ``out`` as CSV."""
    avr = df.groupby([groups]).mean().transpose()
    avr.to_csv(out)
    return avr
