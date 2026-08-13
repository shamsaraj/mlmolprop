# mlmolprop — Cheminformatics & QSAR Modeling Toolkit

**mlmolprop** is a modular Python framework for cheminformatics and QSAR (Quantitative Structure–Activity Relationship) modeling.
It enables molecular descriptor generation, fingerprint computation, dataset preprocessing, feature selection, and machine learning analysis — all within one workflow.

This toolkit integrates **RDKit**, **scikit-learn**, **Keras/PyTorch**, **LIME**, and **matplotlib** to streamline molecular data preparation, model building, and interpretation.

---

## Project Structure

| Module | Description |
|--------|--------------|
| **basic.py** | Core statistics helpers for QSAR model evaluation (PRESS, R², Q², RMSE, F-statistic, etc.). |
| **correlation.py** | Detects and removes highly correlated features from datasets. |
| **processing.py** | Dataset preprocessing — imputation, scaling, normalization, feature selection, correlation filtering, and train/test splitting. |
| **fingerprint.py** | Generates molecular fingerprints (ECFP, MACCS, Avalon, RDKit, etc.) using RDKit. |
| **descriptors.py** | Computes molecular descriptors, merges with activity data, and exports QSAR-ready datasets. |
| **molprep.py** | Prepares molecular structures (SDF/SMILES): salt removal, 3D embedding, charge calculation, and image generation. |
| **model.py** | Fits and evaluates regression/classification models (PLS, Random Forest, SVM, MLP, a small Keras deep-learning classifier, etc.). |
| **importance.py** | Explains model predictions with LIME, and partial dependence plots. |
| **image.py** | Converts molecule/data images to on-disk artifacts (PNG, CSV, SVG). |
| **plots.py** | Scatter/histogram/2D-histogram/ROC grid plots of a feature set vs. a target. |

---

## Installation

Clone the repository:
```bash
git clone https://github.com/shamsaraj/mlmolprop.git
cd mlmolprop
```

Create the environment (conda, recommended):
```bash
conda env create -f environment.yml
conda activate mlmolprop
```

Or install directly with pip:
```bash
pip install -e .
```

> **Note:** the `dl` extra (`M="dl"` in `Model`/`ModelC`) is built on
> Keras 3 running on the PyTorch backend -- set `KERAS_BACKEND=torch`
> before using it. This path is tested and verified to build, train, and
> predict the same way a TensorFlow backend would (see
> `tests/test_model.py::test_modelc_dl_actually_trains`). On Intel
> macOS, `pip install "mlmolprop[dl]"` will pull in PyTorch 2.2.2, the
> newest version PyPI publishes for that platform -- it predates the
> PyTorch API Keras 3's torch backend needs and fails to import.
> conda-forge is recommended for the `dl` extra on that platform
> instead, which has PyTorch 2.13+.

> **Note:** `pip install "mlmolprop[image]"` (needed for
> `svg_files_to_png()` and `RDimage()`) builds `rlPyCairo`'s `pycairo`
> dependency from source against a system Cairo install -- there's no
> prebuilt PyPI wheel on any platform. conda-forge is recommended for
> this extra instead: `conda install -c conda-forge rlpycairo`.

---

## Usage Overview

### 1. Prepare Molecules
```python
from mlmolprop import mol_enumerate

molecules = mol_enumerate("input.sdf", "prepared_3d.sdf", "prepared_2d.sdf")
```

### 2. Generate Descriptors
```python
from mlmolprop import desc, dataframe

desc_data = desc(molecules, source="molecule")
merged = dataframe(desc_data, "activity.csv", "merged.csv")
```

### 3. Process Dataset
```python
from mlmolprop import data_prep

result = data_prep("merged.csv", Scaled="on", FS="reg", Cor="on", mod="reg")
```

### 4. Build and Evaluate Models
```python
from mlmolprop import Model

X_train, y_train, X_test, y_test, v_names = result[:5]
model_result = Model(X_train, y_train, X_test, y_test, v_names, M="rf")
print(model_result[0])  # metrics dict
```

---

## Example Notebook

[`examples/quickstart.ipynb`](examples/quickstart.ipynb) runs the full pipeline
above end-to-end against
[`examples/data/esol_subset.csv`](examples/data/esol_subset.csv), a
70-compound sample of the real, measured **ESOL/Delaney aqueous solubility
dataset**, then goes further:

- molecule preparation → descriptors → dataset processing → model training → evaluation
- algorithm comparison across several regressor types
- a combined hyperparameter + feature-selection grid search, with a heatmap
  of the results
- refit and diagnostics (predicted-vs-observed, residuals) for the model
  that actually won the search, not just an arbitrary starting guess
- feature importance: variable importance + partial dependence
- LIME: explaining a single molecule's prediction
- 2D structure depictions, including substructure highlighting
- a low-dimensional (PCA) map colored by measured activity, and a
  K-means clustering map
- classification, with confusion matrices for the best-by-CV classifier
  and for a small Keras deep-learning model

Open it with `jupyter notebook` after installing (the `notebook`/`nbconvert`
dev tools are included in `environment.yml`).

---

## Features

- Molecular structure preparation (SMILES/SDF), salt removal, 3D embedding
- Descriptor and fingerprint generation (ECFP, MACCS, Avalon, RDKit, etc.)
- Data preprocessing: imputation, normalization, scaling, categorical encoding
- Correlation filtering and variance thresholding
- Machine learning for regression and classification, including a small Keras MLP
- Model interpretation via LIME and partial dependence plots
- Visualization: PCA, ROC, histograms, confusion matrices, clustering

---

## Example Workflow

1. Prepare molecules
2. Generate descriptors
3. Process dataset
4. Train model
5. Interpret results

---

## Author

**J. Shamsara**
[GitHub Profile](https://github.com/shamsaraj)

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
