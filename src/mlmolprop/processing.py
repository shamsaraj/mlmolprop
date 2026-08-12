"""Dataset preprocessing: scaling, feature selection, correlation filtering, splitting."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.model_selection import train_test_split

from .correlation import find_correlation
from .descriptors import CI


def _drop_all_zero_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are entirely zero (treats 0 as missing, temporarily)."""
    df = df.replace(0, np.nan)
    df = df.dropna(how="all", axis=1)
    return df.replace(np.nan, 0)


def data_prep(
    datafile,
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
    """Load a CSV dataset and run it through the standard QSAR preprocessing pipeline.

    Steps: missing-value imputation, row removal by target value, column
    removal, one-hot encoding of categorical columns, then a train/test
    split, followed by (each individually toggleable) zero-variance
    filtering, normalization, scaling, correlation filtering, and
    univariate feature selection, plus an applicability-domain (CI)
    report. The split happens *before* any of those toggleable steps, and
    each one is fit on the training rows only and applied transform-only
    to the test rows -- so no feature- or target-derived statistic (a
    variance, a mean/std, a correlation, an F-test score, ...) is ever
    computed using information from the test set. The prepared
    ``(X, y, ...)`` tuple is also pickled to ``output`` for later reuse.

    Side effects: writes several diagnostic CSVs to the current working
    directory when the corresponding step runs -- ``cor_mat.csv`` (Cor),
    ``Bonferroni.csv`` (FS="clas"), ``dfout.csv``, and
    ``CI_train.csv``/``CI_test.csv``/``CI_Out_train.csv``/``CI_Out_test.csv``.

    Parameters
    ----------
    datafile : str
        Path to the input CSV.
    Scaled, Normal, Sparse : {"on", "off"}
        Apply StandardScaler / Normalizer / MaxAbsScaler to X.
    FS : {"off", "clas", "reg"}
        Univariate feature selection (SelectKBest) for classification or
        regression, keeping the top ``ndes`` features.
    cat_columns : list[str] or None
        Column names to one-hot encode (via ``pandas.get_dummies``)
        before variance filtering. Encoding a new external set
        (``newset="on"``) with categories unseen at training time isn't
        handled -- columns are aligned to ``v_names2`` afterward, so a
        missing dummy column raises a clear KeyError instead of silently
        misaligning.
    Cor : {"on", "off"}
        Drop one of each pair of features whose correlation exceeds 0.9.
    ndes : int, default 6
        Number of features kept by ``FS``.
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
        Test-set fraction for the train/test split.
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
    df = pd.read_csv(datafile, low_memory=False)

    if imputation == "on":
        # NOTE: known-fragile path (was left partially broken in the original
        # code: missing_values="NaN" as a string never matches actual NaN
        # floats, so imputation silently did nothing). Fixed to use np.nan.
        from sklearn.impute import SimpleImputer

        df = df.dropna(axis=1, how="all")
        df1 = df.drop(NAME, axis=1)
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean", copy=True).fit(
            df1
        )
        df2 = imputer.transform(df1)
        df = pd.DataFrame(
            df2, columns=list(df.columns.values[1:]), index=df.loc[:, NAME]
        )

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
    if mod == "class":
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

    if FS in ("clas", "reg"):
        if FS == "clas":
            selector = SelectKBest(f_classif, k=ndes).fit(X_train, y_train)
            p_values = selector.pvalues_
            bonferroni = p_values * len(v_names2)
            bonferroni_df = pd.DataFrame(bonferroni, index=v_names, columns=["p value"])
            bonferroni_df.to_csv("Bonferroni.csv")
        else:
            selector = SelectKBest(score_func=f_regression, k=ndes).fit(
                X_train, y_train
            )
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
