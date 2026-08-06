"""Tests for chemsar.image."""

import os

import numpy as np
import pytest

from chemsar.image import highlight, images_to_dataframe, svg_files_to_png


def test_highlight_returns_valid_svg():
    svg = highlight("[nH]1cnc2cncnc21", "ccn")
    assert "<svg" in svg


def test_images_to_dataframe_round_trips_to_csv(cwd_tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("imgs", exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(3):
        plt.imsave(f"imgs/pic{i}.png", rng.random((4, 4)), cmap="gray")

    df = images_to_dataframe("imgs/", ".png", "out.csv")
    assert df.shape[0] == 3
    assert (cwd_tmp_path / "out.csv").exists()


def test_images_to_dataframe_no_matching_files_raises(cwd_tmp_path):
    os.makedirs("empty_dir", exist_ok=True)
    with pytest.raises(ValueError):
        images_to_dataframe("empty_dir/", ".png", "out.csv")


def test_svg_files_to_png(cwd_tmp_path):
    svg = highlight("c1ccccc1", "c1ccccc1")
    with open("mol.svg", "w") as f:
        f.write(svg)
    svg_files_to_png(["mol.svg"], output_dir=".")
    assert (cwd_tmp_path / "mol.png").exists()
