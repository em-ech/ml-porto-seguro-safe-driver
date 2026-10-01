# Porto Seguro Safe Driver Prediction

Predicting which policyholders will file an auto insurance claim in the next year. Coursework on imbalanced learning for Advanced Machine Learning at IE School of Science and Technology (Master in Computer Science and Business Technology, 2026), built on the Kaggle competition of the same name.

## Data

The data comes from the [Kaggle competition](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction) and is not included.

- 595,212 training rows, of which 21,694 (3.64%) filed a claim
- 892,816 test rows
- 217 features after engineering

## Contents

- `HW4_Porto_Seguro.ipynb`: exploration, missing-value analysis, feature engineering, LightGBM with Bayesian hyperparameter optimization, and evaluation
- `build_notebook.py`: generates the notebook

## Result

Out-of-fold normalized Gini: 0.288.

## Stack

Python, LightGBM, scikit-learn, pandas, NumPy, matplotlib, seaborn.
