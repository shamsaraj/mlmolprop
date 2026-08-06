"""Tests for mlmolprop.correlation."""

import pytest

from mlmolprop.correlation import find_correlation


def test_find_correlation_catches_positive_correlation(feature_frame):
    # f1 and f2 are ~perfectly positively correlated in the fixture
    removed = find_correlation(feature_frame, threshold=0.9)
    assert "f1" in removed or "f2" in removed
    assert not ("f1" in removed and "f2" in removed)  # exactly one survives


def test_find_correlation_catches_negative_correlation(rng):
    import pandas as pd

    n = 100
    a = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a": a,
            "c": -a * 3
            + rng.normal(scale=0.001, size=n),  # ~perfectly negatively correlated
            "d": rng.normal(size=n),
        }
    )
    removed = find_correlation(df, threshold=0.9)
    # regression guard: this used to only compare signed correlation > threshold,
    # so a strong negative correlation was never flagged
    assert "a" in removed or "c" in removed


def test_find_correlation_independent_features_untouched(rng):
    import pandas as pd

    df = pd.DataFrame(rng.normal(size=(50, 3)), columns=["x", "y", "z"])
    assert find_correlation(df, threshold=0.9) == []


def test_find_correlation_invalid_threshold_raises(feature_frame):
    with pytest.raises(ValueError):
        find_correlation(feature_frame, threshold=1.5)
    with pytest.raises(ValueError):
        find_correlation(feature_frame, threshold=0)
