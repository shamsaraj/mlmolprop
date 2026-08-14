"""Detect and select for removal highly correlated numeric features."""

from __future__ import annotations

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
    columns = list(data.columns)
    corr_mat = data.corr().abs()

    # Union-find over pairs whose correlation exceeds threshold, so columns
    # linked only transitively (A~B and B~C both exceed threshold, even if
    # A~C doesn't) end up in the same cluster rather than being resolved as
    # two independent pairs that can each keep a representative that turns
    # out to be correlated with the other pair's representative.
    parent = {c: c for c in columns}

    def find(c: str) -> str:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            if corr_mat.loc[a, b] > threshold:
                union(a, b)

    clusters: dict[str, list[str]] = {}
    for c in columns:
        clusters.setdefault(find(c), []).append(c)

    to_remove: list[str] = []
    for members in clusters.values():
        if len(members) > 1:
            # Keep the last column in original order, matching this
            # function's existing convention for a simple two-column match
            # (the earlier column is removed, the later one kept).
            survivor = members[-1]
            to_remove.extend(m for m in members if m != survivor)

    return to_remove
