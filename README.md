# chemsar — Cheminformatics & QSAR Modeling Toolkit

**chemsar** is a modular Python framework for cheminformatics and QSAR (Quantitative Structure–Activity Relationship) modeling.
It enables molecular descriptor generation, fingerprint computation, dataset preprocessing, feature selection, and machine learning analysis — all within one workflow.

This toolkit integrates **RDKit**, **scikit-learn**, **Keras/TensorFlow**, **LIME**, and **matplotlib** to streamline molecular data preparation, model building, and interpretation.

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
git clone https://github.com/shamsaraj/Chem_ML.git
cd Chem_ML
```

Create the environment (conda, recommended):
```bash
conda env create -f environment.yml
conda activate chemsar
```

Or install directly with pip:
```bash
pip install -e .
```

---

## Usage Overview

### 1. Prepare Molecules
```python
from chemsar import mol_enumerate

molecules = mol_enumerate("input.sdf", "prepared_3d.sdf", "prepared_2d.sdf")
```

### 2. Generate Descriptors
```python
from chemsar import desc, dataframe

desc_data = desc(molecules, source="molecule")
merged = dataframe(desc_data, "activity.csv", "merged.csv")
```

### 3. Process Dataset
```python
from chemsar import data_prep

result = data_prep("merged.csv", Scaled="on", FS="reg", Cor="on", mod="reg")
```

### 4. Build and Evaluate Models
```python
from chemsar import Model

X_train, y_train, X_test, y_test, v_names = result[:5]
model_result = Model(X_train, y_train, X_test, y_test, v_names, M="rf")
print(model_result[0])  # metrics dict
```

---

## Example Notebook

[`examples/quickstart.ipynb`](examples/quickstart.ipynb) runs the full pipeline
above end-to-end against a small set of real molecules with a synthetic
activity value: molecule preparation → descriptors → dataset processing →
model training → evaluation plots. Open it with `jupyter notebook` after
installing (the `notebook`/`nbconvert` dev tools are included in
`environment.yml`).

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
