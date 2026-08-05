"""Grid plots (scatter/histogram/2D-histogram/ROC) of a feature set vs. a target."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

_SMALL_SIZE = 8
_MEDIUM_SIZE = 12
_BIGGER_SIZE = 16


def plot_features(
    df: pd.DataFrame,
    y,
    kind: str,
    grid_size: int = 8,
    df2: pd.DataFrame | None = None,
    y2=None,
    output: str = "plot.jpg",
    show: bool = True,
) -> None:
    """Plot every column of ``df`` against ``y`` in a ``grid_size`` x ``grid_size`` grid.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature matrix; one subplot is drawn per column.
    y : array-like
        Target values, used by "scat", "his2d", and "rocc".
    kind : str
        One of:

        - ``"scat"``: scatter plot of each feature vs. ``y``. If ``df2``
          (and ``y2``) are given, they're overlaid in black.
        - ``"his"``: histogram of each feature. If ``df2`` is given, its
          histogram is overlaid in black.
        - ``"his2d"``: 2D histogram of each feature vs. ``y``.
        - ``"rocc"``: ROC curve treating each feature as a classifier
          score against binary labels ``y``.
    grid_size : int, default 8
        Subplot grid is ``grid_size`` x ``grid_size``.
    df2, y2 : optional
        Second dataset overlaid on "scat"/"his" plots.
    output : str, default "plot.jpg"
        Path the figure is saved to.
    show : bool, default True
        Whether to call ``plt.show()`` after saving.
    """
    rc_params = {
        "font.size": _SMALL_SIZE,
        "axes.titlesize": _SMALL_SIZE,
        "axes.labelsize": _MEDIUM_SIZE,
        "xtick.labelsize": _SMALL_SIZE,
        "ytick.labelsize": _SMALL_SIZE,
        "legend.fontsize": _SMALL_SIZE,
        "figure.titlesize": _BIGGER_SIZE,
    }
    columns = list(df.columns.values)

    with plt.rc_context(rc_params):
        plt.figure()
        for i, column_name in enumerate(columns):
            plt.subplot(grid_size, grid_size, i + 1)

            if kind == "scat":
                plt.title(column_name, fontsize=_MEDIUM_SIZE)
                plt.yticks([1, 2])
                plt.ylabel("Class", fontsize=_SMALL_SIZE)
                plt.xlabel("Value", fontsize=_SMALL_SIZE)
                plt.scatter(df.iloc[:, i], y, color="#e60073", s=500, alpha=0.02)
                if df2 is not None:
                    plt.scatter(df2.iloc[:, i], y2, color="#000000", s=500, alpha=0.02)

            elif kind == "his":
                plt.hist(df.iloc[:, i], color="#e67300", alpha=1)
                if df2 is not None:
                    plt.hist(df2.iloc[:, i], color="#000000", alpha=1)
                plt.title(column_name, fontsize=_MEDIUM_SIZE)
                plt.ylabel("Number", fontsize=_SMALL_SIZE)
                plt.xlabel("", fontsize=_MEDIUM_SIZE)

            elif kind == "his2d":
                plt.hist2d(df.iloc[:, i], y, alpha=1.0, cmap=plt.cm.Greens)
                plt.title(column_name, fontsize=_BIGGER_SIZE)
                plt.ylabel("Class")

            elif kind == "rocc":
                fpr, tpr, _ = metrics.roc_curve(y, df.iloc[:, i], pos_label=1)
                auc = metrics.auc(fpr, tpr)
                plt.plot(
                    fpr,
                    tpr,
                    color="darkorange",
                    lw=2,
                    label=f"ROC curve (area = {auc:0.2f})",
                )
                plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate", fontsize=_SMALL_SIZE)
                plt.ylabel("True Positive Rate", fontsize=_SMALL_SIZE)
                plt.legend(loc="lower right")
                plt.title(column_name, fontsize=_MEDIUM_SIZE)

            else:
                raise ValueError(
                    f"kind must be one of 'scat', 'his', 'his2d', 'rocc', got {kind!r}"
                )

            plt.tight_layout(pad=1, w_pad=0, h_pad=0)

        plt.savefig(output, format="jpg", dpi=2400)
        if show:
            plt.show()
        plt.close()
