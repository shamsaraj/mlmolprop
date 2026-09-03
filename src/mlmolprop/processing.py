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
        if FS == "clas":
            selector = SelectKBest(f_classif, k=ndes).fit(X_train, y_train)
            p_values = selector.pvalues_
            # Must be the feature count SelectKBest actually tested just
            # now (X_train.shape[1]), not v_names2 -- that was fixed back
            # when VarianceThreshold ran, and goes stale if Cor="on" drops
            # correlated features afterward.
            bonferroni = p_values * X_train.shape[1]
            bonferroni_df = pd.DataFrame(bonferroni, index=v_names, columns=["p value"])
            bonferroni_df.to_csv("Bonferroni.csv")
        elif FS == "reg":
            selector = SelectKBest(score_func=f_regression, k=ndes).fit(
                X_train, y_train
            )
        elif FS == "rfe_clas":
            # step=0.1 (drop 10% of remaining features per iteration) instead of
            # RFE's default step=1 -- one-at-a-time elimination means up to
            # ~n_features refits to reach a small ndes, prohibitively slow on
            # wide feature sets (ECFP4 fingerprints are ~2048 columns).
            selector = RFE(
                LogisticRegression(max_iter=1000), n_features_to_select=ndes, step=0.1
            ).fit(X_train, y_train)
        elif FS == "rfe_reg":
            selector = RFE(
                LinearRegression(), n_features_to_select=ndes, step=0.1
            ).fit(X_train, y_train)
        elif FS == "enet_clas":
            # max_features=None (ndes left unset) keeps every feature the L1
            # penalty doesn't zero out, instead of forcing an exact count.
            selector = SelectFromModel(
                LogisticRegression(solver="saga", l1_ratio=0.5, max_iter=5000),
                max_features=ndes,
            ).fit(X_train, y_train)
        elif FS == "enet_reg":
            selector = SelectFromModel(
                ElasticNet(l1_ratio=0.5), max_features=ndes
            ).fit(X_train, y_train)
        else:
            raise ValueError(f"Unknown FS: {FS!r}")
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
