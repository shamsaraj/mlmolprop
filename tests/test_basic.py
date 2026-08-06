"""Tests for chemsar.basic -- core QSAR statistics helpers."""

import numpy as np
import pytest

from chemsar.basic import (
    RMSEP_CV_C,
    F,
    analyse,
    makecolumn,
    press,
    press_m,
    press_root,
    q2r2,
    r2test,
    twodlist,
)


def test_press():
    assert press([1, 2, 3], [1, 2, 4]) == pytest.approx(1.0)


def test_press_length_mismatch_raises():
    with pytest.raises(ValueError):
        press([1, 2], [1, 2, 3])


def test_press_root():
    assert press_root([1, 2, 3], [1, 2, 5]) == pytest.approx(2.0)


def test_press_m_accepts_scalar_or_array_reference():
    values = [1.0, 2.0, 3.0]
    # scalar mean (already computed) and a raw array to average both work
    assert press_m(values, np.mean(values)) == pytest.approx(press_m(values, values))


def test_q2r2_perfect_prediction_is_one():
    obs = [1.0, 2.0, 3.0, 4.0]
    assert q2r2(obs, obs) == pytest.approx(1.0)


def test_q2r2_constant_observed_returns_zero_not_crash():
    assert q2r2([5, 5, 5], [1, 2, 3]) == 0.0


def test_r2test_constant_train_returns_zero_not_crash():
    assert r2test([1, 2, 3], [1, 2, 3], [7, 7, 7]) is not None


def test_rmsep_cv_c_uses_root_not_square():
    # obs=[1,2,3], pred=[1,2,4]: PRESS=1, n=3, MSE=1/3, true RMSE=sqrt(1/3)
    obs, pred = [1, 2, 3], [1, 2, 4]
    rmse = RMSEP_CV_C(obs, pred)
    assert rmse == pytest.approx(np.sqrt(1 / 3))
    # regression guard: this used to be numpy.square(MSE) = (1/3)**2 = 0.111,
    # nowhere near the true RMSE of ~0.577
    assert rmse != pytest.approx(np.square(1 / 3))


def test_rmsep_cv_c_insufficient_degrees_of_freedom_raises():
    with pytest.raises(ValueError):
        RMSEP_CV_C([1, 2, 3], [1, 2, 3], k=5)


def test_f_statistic_matches_manual_msr_over_mse():
    obs = [1.0, 2.0, 3.0, 4.0, 5.0]
    pred = [1.1, 1.9, 3.2, 3.8, 5.3]
    k = 1
    n = len(obs)
    msr = press_m(pred, np.mean(obs)) / k
    mse = press(obs, pred) / (n - k - 1)
    assert F(obs, pred, k) == pytest.approx(msr / mse)


def test_analyse_returns_expected_keys():
    ytrain = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred_train = [1.1, 1.9, 3.2, 3.8, 5.3]
    ytest = [2.0, 3.0]
    y_pred_test = [2.2, 2.7]
    ycv1 = ytrain
    ycv2 = y_pred_train
    result = analyse(ytrain, y_pred_train, ytest, y_pred_test, ycv1, ycv2, k=1)
    expected_keys = {
        "R2",
        "R2_Adj",
        "R2_test",
        "F",
        "q2",
        "RMSE_train",
        "MAE_train",
        "RMSE_test",
        "MAE_test",
        "RMSE_CV",
        "MAE_CV",
    }
    assert set(result.keys()) == expected_keys
    assert all(np.isfinite(v) for v in result.values())


def test_twodlist_shape_and_placeholder():
    grid = twodlist(2, 3)
    assert len(grid) == 2
    assert all(len(row) == 3 for row in grid)
    assert all(cell == "none" for row in grid for cell in row)


def test_makecolumn_extracts_columns():
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    columns = makecolumn(data, 2)
    assert list(columns[0]) == [1, 4, 7]
    assert list(columns[1]) == [2, 5, 8]
