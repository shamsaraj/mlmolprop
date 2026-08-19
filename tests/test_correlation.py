"""Tests for mlmolprop.correlation."""

import numpy as np
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


def test_find_correlation_no_two_survivors_exceed_threshold(rng):
    import pandas as pd

    # (A, Y, B) jointly valid target correlations (checked PSD): A~B and
    # Y~B both above threshold, A~Y comfortably below it.
    target = np.array(
        [
            [1.00, 0.75, 0.93],
            [0.75, 1.00, 0.93],
            [0.93, 0.93, 1.00],
        ]
    )
    n = 4000
    A, Y, B = rng.multivariate_normal(mean=[0, 0, 0], cov=target, size=n).T
    X = A + 0.3 * rng.normal(size=n)  # correlated enough with A to claim it,
    # but not so tightly that X inherits A's own correlation with B above
    # threshold too.
    df = pd.DataFrame({"X": X, "A": A, "Y": Y, "B": B})

    removed = find_correlation(df, threshold=0.9)
    survivors = [c for c in df.columns if c not in removed]
    corr = df[survivors].corr().abs()

    for i, a in enumerate(survivors):
        for b in survivors[i + 1 :]:
            assert corr.loc[a, b] <= 0.9, f"{a} and {b} both survived but corr={corr.loc[a, b]}"


def test_find_correlation_fully_connected_triple_keeps_exactly_one(rng):
    import pandas as pd

    # Edge case: 3 features all pairwise-perfectly correlated (simple
    # single cluster, no order-dependent multi-seed interaction) -- exactly
    # 2 should be removed, not 0 and not 3.
    base = rng.normal(size=200)
    df = pd.DataFrame(
        {
            "a": base,
            "b": base * 2 + rng.normal(scale=0.001, size=200),
            "c": base * -1 + rng.normal(scale=0.001, size=200),
        }
    )
    removed = find_correlation(df, threshold=0.9)
    assert len(removed) == 2
    assert len(set(df.columns) - set(removed)) == 1


def test_find_correlation_scales_to_thousands_of_columns(rng):
    import time

    import pandas as pd

    # Regression guard: the original implementation scanned every pair via
    # DataFrame.loc in a Python loop, ~2 minutes for 2048 columns (a typical
    # ECFP4 fingerprint width). The vectorized implementation should handle
    # this comfortably inside a fraction of that.
    df = pd.DataFrame(rng.integers(0, 2, size=(300, 1500)))
    start = time.time()
    find_correlation(df, threshold=0.9)
    elapsed = time.time() - start
    assert elapsed < 30


def test_find_correlation_known_answer_exact_duplicate(rng):
    import pandas as pd

    # Known-answer case: two independent columns plus a third that's an
    # exact duplicate of the first -- to_remove must contain exactly one
    # name, fully determined by construction. Verified directly which one:
    # the *earlier* column ("x", the processing seed) is the one removed,
    # and the later-discovered match ("x_copy") is what's kept -- the seed
    # is always appended to its own match-group and so always ends up in
    # the "remove everything except group[0]" slice.
    df = pd.DataFrame(
        {
            "x": rng.normal(size=100),
            "y": rng.normal(size=100),
        }
    )
    df["x_copy"] = df["x"]
    removed = find_correlation(df, threshold=0.9)
    assert removed == ["x"]
