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
        selector = _fit_selector(X_train, y_train, FS, ndes)
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


def _elastic_net(kind: str):
    """The elastic-net estimator behind ``FS="enet_clas"``/``"enet_reg"``."""
    if kind == "clas":
        return LogisticRegression(solver="saga", l1_ratio=0.5, max_iter=5000)
    return ElasticNet(l1_ratio=0.5)


def _fit_selector(X_train, y_train, FS: str, ndes):
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
            _elastic_net(FS[5:]), threshold=_NONZERO, max_features=ndes
        ).fit(X_train, y_train)
    raise ValueError(f"Unknown FS: {FS!r}")


_CV_FS = ("clas", "rfe_clas", "enet_clas")


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
        keeps every feature the elastic net doesn't zero out.
    model : unfitted classifier, optional
        Cloned for every fold and configuration. Defaults to
        ``ComplementNB()``.
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

    Returns
    -------
    pandas.DataFrame, or (pandas.DataFrame, pandas.DataFrame) with ``return_folds``
        One row per configuration, in the given order: ``FS``, ``ndes``,
        ``n_features`` (features kept, averaged over folds), and from the
        pooled out-of-fold predictions of each repeat, averaged over the
        repeats: ``F2`` (recall weighted twice as much as precision), ``MCC``,
        ``recall``, ``specificity``, ``precision`` and ``accuracy``. With
        ``return_folds``, also one row per configuration and fold: ``FS``,
        ``ndes``, ``repeat``, ``fold`` (numbered across repeats),
        ``n_features``, the same metrics on the fold's held-out rows, and
        each again on the fold's training rows (``F2_train``,
        ``MCC_train``, ...).
    """
    from sklearn.base import clone
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.naive_bayes import ComplementNB

    configs = [(fs, ndes) for fs, ndes in configs]
    unknown = sorted({fs for fs, _ in configs} - set(_CV_FS))
    if unknown:
        raise ValueError(f"FS must be one of {_CV_FS}, got {unknown}")
    model = ComplementNB() if model is None else model
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    positive = np.unique(y)[-1]

    folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=rs)
    predicted = {c: np.empty((n_repeats, len(y)), dtype=y.dtype) for c in configs}
    kept = {c: [] for c in configs}
    fold_rows = []
    for i, (train, held_out) in enumerate(folds.split(X, y)):
        X_train, X_held, y_train = X.iloc[train], X.iloc[held_out], y[train]
        columns = X_train.columns[VarianceThreshold(vt).fit(X_train).get_support()]
        X_train, X_held = X_train[columns], X_held[columns]
        if cor is not None:
            dropped = find_correlation(X_train, cor)
            X_train, X_held = X_train.drop(columns=dropped), X_held.drop(columns=dropped)
        enet = None  # one elastic-net fit per fold serves every enet setting
        for fs, ndes in configs:
            if fs == "enet_clas":
                enet = enet if enet is not None else _elastic_net("clas").fit(X_train, y_train)
                mask = SelectFromModel(
                    enet, prefit=True, threshold=_NONZERO, max_features=ndes
                ).get_support()
            else:
                mask = _fit_selector(X_train, y_train, fs, ndes).get_support()
            fitted = clone(model).fit(X_train.loc[:, mask], y_train)
            held_pred = fitted.predict(X_held.loc[:, mask])
            predicted[fs, ndes][i // n_splits, held_out] = held_pred
            kept[fs, ndes].append(int(mask.sum()))
            if return_folds:
                train_scores = _class_scores(y_train, fitted.predict(X_train.loc[:, mask]), positive)
                fold_rows.append({
                    "FS": fs, "ndes": ndes, "repeat": i // n_splits, "fold": i,
                    "n_features": int(mask.sum()),
                    **_class_scores(y[held_out], held_pred, positive),
                    **{f"{k}_train": v for k, v in train_scores.items()},
                })

    rows = []
    for fs, ndes in configs:
        per_repeat = pd.DataFrame([_class_scores(y, p, positive) for p in predicted[fs, ndes]])
        rows.append({"FS": fs, "ndes": ndes, "n_features": float(np.mean(kept[fs, ndes])),
                     **per_repeat.mean().to_dict()})
    table = pd.DataFrame(rows)
    return (table, pd.DataFrame(fold_rows)) if return_folds else table


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


def _setting_key(frame: pd.DataFrame) -> pd.Series:
    """One label per (FS, ndes), with ndes=None/NaN as "all"."""
    ndes = frame["ndes"].map(lambda n: "all" if pd.isna(n) else str(int(n)))
    return frame["FS"].astype(str) + " " + ndes


def best_without_overfitting(
    table: pd.DataFrame, folds: pd.DataFrame, metric: str = "MCC", alpha: float = 0.05
) -> pd.DataFrame:
    """Choose a feature-selection setting: the best cross-validated score that does not overfit.

    Takes :func:`feature_selection_cv`'s ``table`` and ``folds``
    (``return_folds=True``). A setting's overfitting is its gap: ``metric``
    on each fold's own training rows minus on its held-out rows, averaged
    over the folds. Every model fits its training rows better than unseen
    ones, so the gaps are compared, not tested against zero:

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
    chosen instead scores about as well.

    Parameters
    ----------
    table, folds : pandas.DataFrame
        :func:`feature_selection_cv` with ``return_folds=True``.
    metric : str, default "MCC"
        Any of its metrics: ``"MCC"``, ``"F2"``, ``"recall"``, ...
    alpha : float, default 0.05
        Significance level of both one-sided tests.

    Returns
    -------
    pandas.DataFrame
        ``table`` sorted by ``metric`` (best first, then fewer features),
        with ``<metric>_train`` (mean over the folds), ``gap``,
        ``p_below_best`` and ``competitive`` (step 1; ``p_below_best`` is
        NaN for the best itself), ``p_overfits`` and ``overfits`` (step 3;
        NaN and False for the reference itself) and ``chosen``, True on
        exactly one row.
    """
    folds = folds.assign(setting=_setting_key(folds), gap=folds[f"{metric}_train"] - folds[metric])
    n_splits = int(folds.groupby("repeat")["fold"].nunique().max())
    held = folds.pivot(index="fold", columns="setting", values=metric)
    gaps = folds.pivot(index="fold", columns="setting", values="gap")

    out = table.assign(setting=_setting_key(table))
    out[f"{metric}_train"] = out["setting"].map(folds.groupby("setting")[f"{metric}_train"].mean())
    out["gap"] = out["setting"].map(gaps.mean())
    out = out.sort_values([metric, "n_features"], ascending=[False, True], kind="stable")

    best = out["setting"].iloc[0]
    out["p_below_best"] = [
        np.nan if s == best else corrected_resampled_ttest(held[best], held[s], n_splits, "greater")[1]
        for s in out["setting"]
    ]
    out["competitive"] = (out["setting"] == best) | (out["p_below_best"] >= alpha)
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
