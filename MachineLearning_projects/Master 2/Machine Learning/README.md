# Heart Failure Risk Prediction — Smurf Society 🧪

**Course:** LELEC2870 – Machine Learning, UCLouvain (Master 2)

## 🎯 Objective

Predict a continuous **clinical heart-failure risk score** for members of the Smurf society from structured health data (age, lifestyle, clinical markers…), with a subset of patients also providing an **image** (multimodal data).

## 🧩 Approach

The notebook is organized in three parts, of increasing model complexity:

- **Part 1 — Linear models:** OLS/SGD variants of Linear Regression, Lasso, Ridge and ElasticNet, on top of a leakage-safe preprocessing pipeline (ordinal/nominal encoding, scaling, imputation)
- **Part 2 — Non-linear models:** Decision Tree, Random Forest, XGBoost, SVR, MLP and KNN, combined with feature selection (Mutual Information, mRMR, Forward Selection) and **Optuna** hyperparameter tuning
- **Part 3 — Multimodal:** a **CNN (PyTorch)** on the image data, then an XGBoost model combining image-derived and tabular features

Each part is evaluated by cross-validated RMSE, with a final error characterization and predictions produced for the unlabeled set.

## 🛠️ Tech stack

pandas, NumPy, scikit-learn, XGBoost, PyTorch / torchvision, Optuna, SciPy, Pillow (PIL), matplotlib / seaborn / plotly

## 📂 Repository contents

| File / Folder | Description |
|---|---|
| `Notebook.ipynb` | Full pipeline: data cleaning → EDA → Part 1/2/3 modeling → evaluation |
| `data_labeled/` | Labeled training data (tabular + images) |
| `LELEC2870___Machine_learning_FinalSubmission.pdf` | Final report |
| `project_guidelines.pdf` | Original assignment brief |
