# Chem_ML — Cheminformatics & QSAR Modeling Toolkit

**Chem_ML** is a modular Python framework for cheminformatics and QSAR (Quantitative Structure–Activity Relationship) modeling.  
It enables molecular descriptor generation, fingerprint computation, dataset preprocessing, feature selection, and machine learning analysis — all within one workflow.

This toolkit integrates **RDKit**, **Scikit-learn**, **LIME**, and **matplotlib** to streamline molecular data preparation, model building, and interpretation.

---

## Project Structure

| File | Description |
|------|--------------|
| **basic.py** | Core mathematical/statistical helper functions (PRESS, R², RMSE, F-test, etc.). |
| **correlation.py** | Detects and removes highly correlated features from datasets. |
| **processing.py** | Handles dataset preprocessing — scaling, normalization, feature selection, correlation filtering, and dataset splitting. |
| **fingerprint.py** | Generates molecular fingerprints (ECFP, MACCS, Avalon, RDKit, etc.) using RDKit. |
| **descriptors.py** | Computes molecular descriptors, merges with activity data, and exports QSAR-ready datasets. |
| **molprep.py** | Prepares molecular structures (SDF/SMILES): 3D embedding, salt removal, charge calculation, and image generation. |
| **model.py** | Builds regression and classification models (PLS, Random Forest, SVM, Neural Networks, Deep Learning, etc.) and evaluates their performance. |
| **importance.py** | Explains model predictions using interpretability methods (LIME, TreeInterpreter, Partial Dependence). |
| **image.py** | Processes and augments molecular structure images and converts them into numerical datasets. |
| **plots.py** | Creates histograms, scatter plots, ROC curves, and other visualization outputs. |

---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/shamsaraj/Chem_ML.git
cd Chem_ML
```

Install the dependencies:
```bash

pip install rdkit scikit-learn matplotlib numpy pandas lime keras svglib
```

---

## Usage Overview

### 1️⃣ Prepare Molecules
```python
from modules.molprep import mol_enumerate
molecules = mol_enumerate("input.sdf", "prepared_3d.sdf", "prepared_2d.sdf")
```

### 2️⃣ Generate Descriptors or Fingerprints
```python
from modules.descriptors import desc, dataframe
desc_data = desc("prepared_3d.sdf", type="sdf")
merged = dataframe(desc_data, "activity.csv", "merged.csv")
```

### 3️⃣ Process Dataset
```python
from modules.processing import data_prep
set = data_prep("merged.csv", Scaled="on", FS="reg", Cor="on", mod="reg")
```

### 4️⃣ Build and Evaluate Models
```python
from modules.model import Model
result = Model(set[0], set[1], set[2], set[3], set[4], M="rf")
print(result[0])  # model summary and metrics
```

---

## Features

✅ Molecular structure preparation (SMILES/SDF)  
✅ Descriptor and fingerprint generation (ECFP, MACCS, Avalon, etc.)  
✅ Data preprocessing, normalization, and feature selection  
✅ Correlation filtering and variance thresholding  
✅ Machine learning for regression and classification  
✅ Model interpretation (LIME, Skater, TreeInterpreter)  
✅ Visualization tools: PCA, ROC, histograms, clustering  

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
🔗 [GitHub Profile](https://github.com/shamsaraj)

---

## 🧾 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
