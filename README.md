# mlmolprop — Cheminformatics & QSAR Modeling Toolkit

[![PyPI](https://img.shields.io/pypi/v/mlmolprop)](https://pypi.org/project/mlmolprop/)
[![Python](https://img.shields.io/pypi/pyversions/mlmolprop)](https://pypi.org/project/mlmolprop/)
[![Tests](https://github.com/shamsaraj/mlmolprop/actions/workflows/tests.yml/badge.svg)](https://github.com/shamsaraj/mlmolprop/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/shamsaraj/mlmolprop/blob/main/LICENSE)

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
| **importance.py** | Variable importance for any fitted model in one table format (native, permutation, SHAP, LIME, bit flip, plus model-free enrichment/significance), bootstrap stability, stable-feature selection and cross-method consensus; LIME explanations and partial dependence plots. |
| **image.py** | Converts molecule/data images to on-disk artifacts (PNG, CSV, SVG). |
| **plots.py** | Scatter/histogram/2D-histogram/ROC grid plots of a feature set vs. a target. |

---

## Installation

```bash
pip install mlmolprop
```

Optional extras:

```bash
pip install "mlmolprop[dl]"       # Keras/PyTorch models (M="dl", ModelMT, ...)
pip install "mlmolprop[xgboost]"  # M="xgb"
pip install "mlmolprop[shap]"     # SHAP importance for tree and other non-linear models
pip install "mlmolprop[all]"      # everything installable by pip alone
```

`[all]` deliberately excludes the `image` extra: `rlPyCairo` pulls in `pycairo`,
which publishes binary wheels for Windows only and otherwise compiles against a
system Cairo. Ask for it explicitly, or take it from conda-forge:

```bash
pip install "mlmolprop[image]"          # needs system Cairo + pkg-config
conda install -c conda-forge rlpycairo  # recommended instead
```

For development, or to reproduce the tested environment exactly:

```bash
git clone https://github.com/shamsaraj/mlmolprop.git
cd mlmolprop
conda env create -f environment.yml
conda activate mlmolprop
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

### 5. Interpret the Model
```python
from mlmolprop import (
    ModelC, bootstrap_importance, consensus_features, enrichment_importance,
    stable_features, variable_importance,
)

# A classifier on fingerprint bits (split with data_prep(..., mod="class"))
result, model = ModelC(X_train, y_train, X_test, y_test, v_names, M="cnb")

# One table format whatever the algorithm or method
native = variable_importance(model, v_names)  # coefficients, NB log-odds, tree importances
shap_table = variable_importance(model, v_names, "shap", X=X_train)
lime_table = variable_importance(model, v_names, "lime", X=X_train)

# Stability: refit on bootstrap resamples; keep features whose signal clears its noise
stability = bootstrap_importance(model, X_train, y_train, v_names, n_boot=100)
stable = stable_features(stability, threshold=2.0, reference=native,
                         X=X_train, y=y_train, min_support=3)

# Consensus: features most methods rank in their top 35, per direction
consensus = consensus_features(
    {"native": native, "shap": shap_table, "lime": lime_table, "stability": stability,
     "enrichment": enrichment_importance(X_train, y_train, v_names)},
    top_n=35, min_agree=3,
)
# Each method's top 35 per side is filled even when few features matter, so
# intersect with the stable set rather than reading the consensus alone
final = consensus[consensus["selected"]].index.intersection(stable.index)
```

---

## Example Notebook

[`examples/quickstart.ipynb`](https://github.com/shamsaraj/mlmolprop/blob/main/examples/quickstart.ipynb) runs the full pipeline
above end-to-end against
[`examples/data/esol_subset.csv`](https://github.com/shamsaraj/mlmolprop/blob/main/examples/data/esol_subset.csv), a
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

> **Verified against mlmolprop 1.1.0** -- every cell, including the
> deep-learning classification cell (`M="dl"`), runs end-to-end with no
> errors (`jupyter nbconvert --execute`). The DL cell needs the `dl`
> extra installed; the notebook sets `KERAS_BACKEND=torch` itself, so no
> manual setup is needed beyond `pip install "mlmolprop[dl]"` (or
> conda-forge on Intel macOS -- see Installation above).

Open it with `jupyter notebook` after installing (the `notebook`/`nbconvert`
dev tools are included in `environment.yml`).

---

## Features

- Molecular structure preparation (SMILES/SDF), salt removal, 3D embedding
- Descriptor and fingerprint generation (ECFP, MACCS, Avalon, RDKit, etc.)
- Data preprocessing: imputation, normalization, scaling, categorical encoding
- Correlation filtering and variance thresholding
- Machine learning for regression and classification, including a small Keras MLP
- Model interpretation: variable importance across algorithms and methods, bootstrap
  stability, stable-feature selection and cross-method consensus; LIME and partial
  dependence plots
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

This project is licensed under the **MIT License** — see the [LICENSE](https://github.com/shamsaraj/mlmolprop/blob/main/LICENSE) file for details.
