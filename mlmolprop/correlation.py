"""Detect and select for removal highly correlated numeric features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def find_correlation(data: pd.DataFrame, threshold: float = 0.9) -> list[str]:
    """Find features that are redundant due to high pairwise correlation.

    For every pair of features whose absolute correlation exceeds
    ``threshold``, one feature from the pair is kept and the rest are
    marked for removal, so that at most one representative of each
    correlated cluster survives.

    Parameters
    ----------
    data : pandas.DataFrame
        Numeric feature matrix (columns are features).
    threshold : float, default 0.9
        Absolute correlation threshold in ``(0, 1]``; pairs whose absolute
        correlation exceeds this value are considered redundant.

    Returns
    -------
    list[str]
        Column names recommended for removal.
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    data = pd.DataFrame(data)
    corr_mat = data.corr().abs()
    lower_triangle = pd.DataFrame(
        np.tril(corr_mat, k=-1), index=corr_mat.index, columns=corr_mat.columns
    )

    already_flagged: set[str] = set()
    to_remove: list[str] = []
    for column in lower_triangle.columns:
        group = lower_triangle.index[lower_triangle[column] > threshold].tolist()
        if group and column not in already_flagged:
            already_flagged.update(group)
            group.append(column)
            to_remove.extend(group[1:])

    return to_remove
