"""Tests for mlmolprop.plots."""

import pytest

from mlmolprop.plots import plot_features


@pytest.fixture
def plot_data(feature_frame, rng, cwd_tmp_path):
    y_class = rng.integers(1, 3, size=len(feature_frame))
    y_binary = rng.integers(0, 2, size=len(feature_frame))
    df2 = feature_frame.iloc[:15]
    y2 = rng.integers(1, 3, size=15)
    return feature_frame, y_class, y_binary, df2, y2


@pytest.mark.parametrize("kind", ["scat", "his", "his2d"])
def test_plot_features_valid_kinds(plot_data, kind, cwd_tmp_path):
    df, y_class, y_binary, df2, y2 = plot_data
    plot_features(df, y_class, kind, grid_size=2, output=f"{kind}.jpg", show=False)
    assert (cwd_tmp_path / f"{kind}.jpg").exists()


def test_plot_features_rocc(plot_data, cwd_tmp_path):
    df, y_class, y_binary, df2, y2 = plot_data
    plot_features(df, y_binary, "rocc", grid_size=2, output="rocc.jpg", show=False)
    assert (cwd_tmp_path / "rocc.jpg").exists()


def test_plot_features_his_without_df2_does_not_crash(plot_data, cwd_tmp_path):
    # regression guard: `if "test"=="test":` was always true, so the "his"
    # branch unconditionally tried df2.iloc[...] even when df2 defaulted to
    # "" -- crashing on every default (no-df2) call
    df, y_class, y_binary, df2, y2 = plot_data
    plot_features(df, y_class, "his", grid_size=2, output="his.jpg", show=False)


def test_plot_features_overlay_with_df2(plot_data, cwd_tmp_path):
    df, y_class, y_binary, df2, y2 = plot_data
    plot_features(
        df, y_class, "scat", grid_size=2, df2=df2, y2=y2, output="scat2.jpg", show=False
    )
    plot_features(
        df, y_class, "his", grid_size=2, df2=df2, output="his2.jpg", show=False
    )


def test_plot_features_invalid_kind_raises(plot_data):
    df, y_class, y_binary, df2, y2 = plot_data
    with pytest.raises(ValueError):
        plot_features(df, y_class, "bogus", grid_size=2, show=False)
