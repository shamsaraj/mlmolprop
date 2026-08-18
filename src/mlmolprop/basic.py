"""Core statistics helpers used for QSAR model evaluation."""

from __future__ import annotations

import numpy as np


def _check_same_length(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"inputs must be the same length, got {len(a)} and {len(b)}")


def press(observed, predicted) -> float:
    """Sum of squared residuals (a.k.a. PRESS/RSS) between two sequences."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    _check_same_length(observed, predicted)
    return float(np.sum((observed - predicted) ** 2))


def press_root(observed, predicted) -> float:
    """Sum of absolute residuals between two sequences."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    _check_same_length(observed, predicted)
    return float(np.sum(np.abs(observed - predicted)))


def press_m(values, reference) -> float:
    """Sum of squared deviations of ``values`` from ``mean(reference)``.

    ``reference`` may be a precomputed scalar mean (``mean`` of a scalar is
    itself, so this is a no-op) or an array-like whose mean is used as the
    reference point. Both call styles are used elsewhere in this module.
    """
    values = np.asarray(values, dtype=float)
    reference_mean = np.mean(reference)
    return float(np.sum((values - reference_mean) ** 2))


def q2r2(obs, pred) -> float:
    """Q2/R2 = 1 - PRESS / total sum of squares of ``obs``."""
    obs = np.asarray(obs, dtype=float)
    ss_tot = press_m(obs, np.mean(obs))
    if ss_tot == 0:
        return 0.0
    return 1 - press(obs, pred) / ss_tot


def r2test(obs, pred, train) -> float:
    """External R2 of a test set, using the training set's mean as reference."""
    ss_tot = press_m(obs, train)
    if ss_tot == 0:
        return 0.0
    return 1 - press(obs, pred) / ss_tot


def RMSEP_CV_C(obs, pred, k: int = 0) -> float:
    """Root mean square error, optionally adjusted for degrees of freedom.

    With ``k=0`` this is the plain RMSE (external test set or LOO-CV set).
    With ``k>0`` the denominator is adjusted by the number of predictors/
    components ``k`` (training-set calibration RMSE, a.k.a. SEE).
    """
    ssr = press(obs, pred)
    n = len(obs)
    denom = n if k == 0 else n - k - 1
    if denom <= 0:
        raise ValueError(f"not enough observations ({n}) for k={k} predictors")
    return float(np.sqrt(ssr / denom))


def F(obs, pred, k: int) -> float:
    """F-statistic (MSR / MSE) for a training-set regression with k predictors."""
    obs = np.asarray(obs, dtype=float)
    n = len(obs)
    denom = n - k - 1
    if denom <= 0:
        raise ValueError(f"not enough observations ({n}) for k={k} predictors")
    msr = press_m(pred, np.mean(obs)) / k
    mse = press(obs, pred) / denom
    return msr / mse


def analyse(ytrain, y_pred_train, ytest, y_pred_test, ycv1, ycv2, k: int) -> dict:
    """Compute a standard set of QSAR model-quality metrics.

    Parameters
    ----------
    ytrain, y_pred_train : array-like
        Observed and predicted values on the training set.
    ytest, y_pred_test : array-like
        Observed and predicted values on the external test set.
    ycv1, ycv2 : array-like
        Observed and predicted values under cross-validation.
    k : int
        Number of predictors (or latent components). The cross-validation
        formula used here assumes leave-one-out CV; it is not a valid
        adjustment for k-fold CV.

    Returns
    -------
    dict
        R2, R2_Adj, R2_test, F, q2, RMSE/MAE for train, test, and CV.
    """
    n_train = len(ytrain)
    r2 = q2r2(ytrain, y_pred_train)
    # Both R2_Adj and F encode classical OLS degrees-of-freedom (n > k+1, k
    # literally-fit linear parameters) -- meaningful for M in {mlr, pls,
    # lasso, ...}, but not a real constraint for RF/SVM/tree-style
    # regressors, whose capacity isn't controlled by feature count the same
    # way. Reporting NaN here instead of raising/dividing-by-zero mirrors
    # ModelC(), which has no such constraint, and lets those model types run
    # on any feature count.
    if n_train - k - 1 > 0:
        r2_adj = 1 - (((n_train - 1) / (n_train - k - 1)) * (1 - r2))
    else:
        r2_adj = float("nan")
    q2 = q2r2(ycv1, ycv2)
    r2_test = r2test(ytest, y_pred_test, ytrain)
    try:
        f = F(ytrain, y_pred_train, k)
    except ValueError:
        f = float("nan")
    rmse_train = RMSEP_CV_C(ytrain, y_pred_train)
    mae_train = press_root(ytrain, y_pred_train) / n_train
    rmse_cv = RMSEP_CV_C(ycv1, ycv2)
    mae_cv = press_root(ycv1, ycv2) / len(ycv2)
    rmse_test = RMSEP_CV_C(ytest, y_pred_test)
    mae_test = press_root(ytest, y_pred_test) / len(ytest)
    return {
        "R2": r2,
        "R2_Adj": r2_adj,
        "R2_test": r2_test,
        "F": f,
        "q2": q2,
        "RMSE_train": rmse_train,
        "MAE_train": mae_train,
        "RMSE_test": rmse_test,
        "MAE_test": mae_test,
        "RMSE_CV": rmse_cv,
        "MAE_CV": mae_cv,
    }


def twodlist(m: int, n: int) -> list[list[str]]:
    """Build an m x n list of lists pre-filled with the placeholder "none"."""
    return [["none"] * n for _ in range(m)]


def makecolumn(data, c: int) -> list[list]:
    """Return the first ``c`` columns of ``data`` as a list of Python lists."""
    arr = np.asarray(data)
    return [list(arr[:, j]) for j in range(c)]
