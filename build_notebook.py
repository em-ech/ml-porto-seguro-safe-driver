"""Generates HW4_Porto_Seguro.ipynb from a list of cells.

Keeping the notebook in a generator keeps the source diff-friendly and avoids
hand-editing raw .ipynb JSON. Run:  python build_notebook.py
"""
import json
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# Homework 4 — Imbalanced Learning
## Porto Seguro's Safe Driver Prediction (Kaggle)

**Goal.** Predict the probability that an auto-insurance policy holder files a claim
in the next year. The target is highly imbalanced (~3.6% positives).

**Scoring metric.** Normalized Gini Coefficient, equivalent to `2 * AUC - 1`.

**Pipeline implemented in this notebook**
1. Exploratory analysis — class imbalance, missingness, per-feature target distribution, correlations.
2. Feature engineering — `num_missing`, `mix_ind`, count encoding, one-hot encoding of categoricals, sparse-matrix assembly, and a justified feature-removal step.
3. Modelling — LightGBM with Stratified K-Fold cross-validation, a custom Gini evaluation metric, Bayesian hyper-parameter optimization, and early stopping.
4. Prediction — fold-averaged test probabilities written to `submission.csv` in the Kaggle format.
""")

# ---------------------------------------------------------------------------
md("## 1. Setup")

code(r"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
print("LightGBM", lgb.__version__)
""")

code(r"""
# ----------------------------- Configuration -----------------------------
DATA_DIR = "/Users/emecheverria/Downloads/porto-seguro-safe-driver-prediction"
OUT_DIR = "."

# SAMPLE_FRAC < 1.0 runs a fast smoke test on a stratified subsample.
# Set to 1.0 for the full, submission-quality run.
SAMPLE_FRAC = 1.0

N_FOLDS = 5                 # Stratified K-Fold splits

# Bayesian optimization budget (total evaluations = init + iter).
BO_INIT_POINTS = 4
BO_N_ITER = 8
BO_FOLDS = 3               # cheaper CV used only inside the search
BO_LR = 0.05               # higher LR + early stopping keeps the search fast
BO_MAX_ROUNDS = 1500
BO_EARLY_STOP = 50

# Final model: low learning rate + many rounds, as recommended for convergence.
FINAL_LR = 0.01
FINAL_MAX_ROUNDS = 8000
FINAL_EARLY_STOP = 200

# scale_pos_weight handles imbalance in the gradient. It mainly affects score
# *calibration*, not ranking (Gini is rank-based), so it is optional here.
USE_SCALE_POS_WEIGHT = False
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Scoring metric — Normalized Gini

The exact implementation from the assignment brief. `eval_gini` returns the
normalized Gini coefficient; `gini_lgb` wraps it as a LightGBM custom metric
(`is_higher_better=True`).
""")

code(r"""
def eval_gini(y_true, y_pred):
    '''Normalized Gini coefficient (assignment reference implementation).'''
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    L_mid = np.linspace(1 / n_samples, 1, n_samples)

    # Gini for the predicted ordering
    pred_order = y_true[y_pred.argsort()]
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    G_pred = np.sum(L_mid - L_pred)

    # Gini for a perfect ordering
    true_order = y_true[y_true.argsort()]
    L_true = np.cumsum(true_order) / np.sum(true_order)
    G_true = np.sum(L_mid - L_true)

    return G_pred / G_true


def gini_lgb(y_pred, dataset):
    '''LightGBM feval wrapper: returns (name, value, is_higher_better).'''
    y_true = dataset.get_label()
    return "gini", eval_gini(y_true, y_pred), True
""")

# ---------------------------------------------------------------------------
md("## 3. Load data")

code(r"""
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

if SAMPLE_FRAC < 1.0:
    from sklearn.model_selection import train_test_split
    _, train = train_test_split(
        train, test_size=SAMPLE_FRAC,
        stratify=train["target"], random_state=RANDOM_STATE)
    train = train.reset_index(drop=True)
    print(f"Sub-sampled train to {len(train):,} rows (SAMPLE_FRAC={SAMPLE_FRAC}).")

print("train:", train.shape, "| test:", test.shape)
train.head()
""")

code(r"""
feature_cols = [c for c in train.columns if c not in ("id", "target")]
bin_features = [c for c in feature_cols if c.endswith("_bin")]
cat_features = [c for c in feature_cols if c.endswith("_cat")]
num_features = [c for c in feature_cols if c not in bin_features + cat_features]
ind_features = [c for c in feature_cols if c.startswith("ps_ind")]
calc_features = [c for c in feature_cols if c.startswith("ps_calc")]

print(f"{len(feature_cols)} features: "
      f"{len(bin_features)} binary, {len(cat_features)} categorical, "
      f"{len(num_features)} continuous/ordinal")
print(f"{len(calc_features)} are ps_calc_* features")
""")

# ---------------------------------------------------------------------------
md("## 4. Exploratory analysis")

md("### 4.1 Class imbalance")

code(r"""
pos_rate = train["target"].mean()
n_pos = int(train["target"].sum())
n_neg = len(train) - n_pos
scale_pos_weight = n_neg / n_pos
print(f"Positives: {n_pos:,} ({pos_rate:.2%}) | Negatives: {n_neg:,}")
print(f"Negative/positive ratio (scale_pos_weight): {scale_pos_weight:.2f}")

ax = train["target"].value_counts().sort_index().plot(
    kind="bar", color=["#4C72B0", "#C44E52"], figsize=(4, 3))
ax.set_xticklabels(["0 (no claim)", "1 (claim)"], rotation=0)
ax.set_title("Target class distribution")
ax.set_ylabel("count")
plt.tight_layout(); plt.show()
""")

md("### 4.2 Missing values (`-1` sentinel)")

code(r"""
missing_share = (train[feature_cols] == -1).mean()
missing_share = missing_share[missing_share > 0].sort_values(ascending=False)

ax = missing_share.plot(kind="barh", figsize=(6, 4), color="#55A868")
ax.set_title("Share of missing (-1) values per feature")
ax.set_xlabel("fraction missing")
plt.tight_layout(); plt.show()
missing_share.to_frame("missing_fraction").style.format("{:.2%}")
""")

md(r"""
A handful of features carry most of the missingness (`ps_car_03_cat` ~69%,
`ps_car_05_cat` ~45%, `ps_reg_03` ~18%). The **pattern** of missingness can
itself be predictive, motivating the `num_missing` feature below.
""")

md("### 4.3 Per-feature target distribution")

code(r"""
def plot_target_proportion(cols, axes):
    '''Bar of P(target=1 | feature value) with 95% CI; flat bars => low signal.'''
    for col, ax in zip(cols, axes):
        grp = train.groupby(col)["target"].agg(["mean", "count"])
        se = np.sqrt(grp["mean"] * (1 - grp["mean"]) / grp["count"])
        ax.bar(grp.index.astype(str), grp["mean"], yerr=1.96 * se,
               color="#4C72B0", capsize=3)
        ax.axhline(pos_rate, color="red", ls="--", lw=1, label="base rate")
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)

examples = ["ps_ind_04_cat", "ps_ind_17_bin", "ps_car_07_cat",
            "ps_calc_16_bin", "ps_calc_19_bin", "ps_calc_01"]
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
plot_target_proportion(examples, axes.ravel())
fig.suptitle("Target proportion across feature values "
             "(top: informative, bottom: ps_calc_* are flat)", y=1.02)
plt.tight_layout(); plt.show()
""")

code(r"""
# Quantify signal: spread = max - min of P(target=1|value) for low-cardinality features.
spreads = {}
for col in feature_cols:
    if train[col].nunique() <= 12:
        g = train.groupby(col)["target"].mean()
        spreads[col] = g.max() - g.min()
spread_s = pd.Series(spreads).sort_values()
print("Lowest-signal features (smallest target-proportion spread):")
print(spread_s.head(12).to_string())
""")

md(r"""
The `ps_calc_*` binary features show essentially no variation in target
proportion (spread < 0.001) — they are noise. This is consistent with the
assignment's guidance to drop the `calc` block.
""")

md("### 4.4 Correlation among continuous features")

code(r"""
cont = [c for c in num_features if c not in ("ps_ind_14",)]
corr = train[cont].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
            cbar_kws={"shrink": 0.7})
plt.title("Correlation — continuous/ordinal features")
plt.tight_layout(); plt.show()

high = (corr.abs().where(~np.eye(len(corr), dtype=bool))
        .stack().sort_values(ascending=False))
high = high[high > 0.5].drop_duplicates()
print("Pairs with |corr| > 0.5:")
print(high.to_string())
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Feature engineering

Implemented per the assignment hints:

| Step | What |
|------|------|
| `num_missing` | per-row count of `-1` sentinels |
| `mix_ind` | underscore-concatenation of all `ps_ind_*` features (interaction string) |
| count encoding | frequency of each value for every categorical + `mix_ind` (`*_count`) |
| one-hot | `OneHotEncoder` on `*_cat` features → sparse |
| removal | drop `ps_calc_*`, rare `ps_ind_1x_bin`, `ps_ind_14`, `ps_car_14` |
| assembly | `scipy.sparse.hstack` of numeric block + one-hot block |
""")

code(r"""
# Features to remove (justified by EDA above and the assignment brief):
#   - all ps_calc_*            : no discriminative power (flat target proportion)
#   - ps_ind_10..13_bin        : extremely rare, statistically unreliable (wide CIs)
#   - ps_ind_14                : low signal / noisy
#   - ps_car_14                : redundant / correlated, heavy missingness
DROP_FEATURES = (
    calc_features
    + ["ps_ind_10_bin", "ps_ind_11_bin", "ps_ind_12_bin", "ps_ind_13_bin"]
    + ["ps_ind_14", "ps_car_14"]
)
print(f"Dropping {len(DROP_FEATURES)} features.")
""")

code(r"""
def engineer_features(train_df, test_df):
    '''Build sparse train/test design matrices.

    Train and test are processed together so that count encodings, one-hot
    categories, and the mix_ind vocabulary are consistent across both.
    '''
    n_train = len(train_df)
    df = pd.concat([train_df.drop(columns=["target"]), test_df],
                   axis=0, ignore_index=True)

    # (1) num_missing — computed over the ORIGINAL feature set, before removal.
    df["num_missing"] = (df[feature_cols] == -1).sum(axis=1)

    # (2) mix_ind — interaction string over all indicator features.
    df["mix_ind"] = df[ind_features].astype(str).agg("_".join, axis=1)

    # (3) Count encoding for every categorical feature plus mix_ind.
    count_cols = []
    for col in cat_features + ["mix_ind"]:
        counts = df[col].value_counts()
        df[col + "_count"] = df[col].map(counts).astype(np.float32)
        count_cols.append(col + "_count")

    # (4) Numeric block: kept continuous/ordinal + binary + engineered counts
    #     + num_missing. (Categoricals go through one-hot instead.)
    keep_num = [c for c in num_features + bin_features if c not in DROP_FEATURES]
    numeric_cols = keep_num + count_cols + ["num_missing"]
    X_numeric = sparse.csr_matrix(df[numeric_cols].astype(np.float32).values)

    # (5) One-hot encoding of categorical features (sparse).
    ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    X_cat = ohe.fit_transform(df[cat_features])

    # (6) Assemble.
    X_all = sparse.hstack([X_numeric, X_cat], format="csr")
    feature_names = (numeric_cols
                     + list(ohe.get_feature_names_out(cat_features)))

    X_tr = X_all[:n_train]
    X_te = X_all[n_train:]
    return X_tr, X_te, feature_names


X, X_test, feat_names = engineer_features(train, test)
y = train["target"].values
test_ids = test["id"].values

density = X.nnz / (X.shape[0] * X.shape[1])
print(f"X (train): {X.shape} | X_test: {X_test.shape}")
print(f"Total engineered columns: {len(feat_names)} | density: {density:.3f}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Modelling — LightGBM

Strategy: Stratified K-Fold CV (preserves the 3.6% positive rate per fold),
custom Gini metric, early stopping on the validation fold, and Bayesian
optimization over the hyper-parameter ranges suggested in the brief.
""")

code(r"""
BASE_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "None",          # we drive early stopping with the custom Gini feval
    "verbosity": -1,
    "seed": RANDOM_STATE,
    "feature_pre_filter": False,
}
if USE_SCALE_POS_WEIGHT:
    BASE_PARAMS["scale_pos_weight"] = scale_pos_weight


def cv_gini(params, n_folds, learning_rate, max_rounds, early_stop, seed=RANDOM_STATE):
    '''Out-of-fold normalized Gini for a parameter set.'''
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    p = {**BASE_PARAMS, **params, "learning_rate": learning_rate}
    for tr_idx, va_idx in skf.split(X, y):
        dtr = lgb.Dataset(X[tr_idx], y[tr_idx])
        dva = lgb.Dataset(X[va_idx], y[va_idx], reference=dtr)
        model = lgb.train(
            p, dtr, num_boost_round=max_rounds,
            valid_sets=[dva], feval=gini_lgb,
            callbacks=[lgb.early_stopping(early_stop, verbose=False)],
        )
        oof[va_idx] = model.predict(X[va_idx], num_iteration=model.best_iteration)
    return eval_gini(y, oof)
""")

md("### 6.1 Bayesian hyper-parameter optimization")

code(r"""
PBOUNDS = {
    "num_leaves": (30, 40),
    "lambda_l1": (0.7, 0.9),
    "lambda_l2": (0.9, 1.0),
    "feature_fraction": (0.6, 0.7),
    "bagging_fraction": (0.6, 0.9),
    "min_child_samples": (6, 10),
    "min_child_weight": (10, 40),
}


def lgb_objective(num_leaves, lambda_l1, lambda_l2, feature_fraction,
                  bagging_fraction, min_child_samples, min_child_weight):
    params = {
        "num_leaves": int(round(num_leaves)),
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": 1,
        "min_child_samples": int(round(min_child_samples)),
        "min_child_weight": min_child_weight,
    }
    return cv_gini(params, BO_FOLDS, BO_LR, BO_MAX_ROUNDS, BO_EARLY_STOP)


optimizer = BayesianOptimization(
    f=lgb_objective, pbounds=PBOUNDS, random_state=RANDOM_STATE, verbose=2)
optimizer.maximize(init_points=BO_INIT_POINTS, n_iter=BO_N_ITER)
print("\nBest CV Gini:", round(optimizer.max["target"], 5))
print("Best params:", optimizer.max["params"])
""")

code(r"""
best = optimizer.max["params"]
best_params = {
    "num_leaves": int(round(best["num_leaves"])),
    "lambda_l1": best["lambda_l1"],
    "lambda_l2": best["lambda_l2"],
    "feature_fraction": best["feature_fraction"],
    "bagging_fraction": best["bagging_fraction"],
    "bagging_freq": 1,
    "min_child_samples": int(round(best["min_child_samples"])),
    "min_child_weight": best["min_child_weight"],
}
final_params = {**BASE_PARAMS, **best_params, "learning_rate": FINAL_LR}
final_params
""")

md("### 6.2 Final training with fold-averaged test predictions")

code(r"""
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_pred = np.zeros(len(y))
test_pred = np.zeros(X_test.shape[0])
fold_gini = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    dtr = lgb.Dataset(X[tr_idx], y[tr_idx])
    dva = lgb.Dataset(X[va_idx], y[va_idx], reference=dtr)
    model = lgb.train(
        final_params, dtr, num_boost_round=FINAL_MAX_ROUNDS,
        valid_sets=[dtr, dva], valid_names=["train", "valid"],
        feval=gini_lgb,
        callbacks=[
            lgb.early_stopping(FINAL_EARLY_STOP, verbose=False),
            lgb.log_evaluation(500),
        ],
    )
    oof_pred[va_idx] = model.predict(X[va_idx], num_iteration=model.best_iteration)
    test_pred += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
    g = eval_gini(y[va_idx], oof_pred[va_idx])
    fold_gini.append(g)
    print(f"Fold {fold}: best_iter={model.best_iteration:4d}  Gini={g:.5f}")

print(f"\nMean fold Gini: {np.mean(fold_gini):.5f} +/- {np.std(fold_gini):.5f}")
print(f"Overall OOF Gini: {eval_gini(y, oof_pred):.5f}")
""")

md("### 6.3 Feature importance")

code(r"""
imp = pd.Series(model.feature_importance(importance_type="gain"),
                index=feat_names).sort_values(ascending=False)
ax = imp.head(25).iloc[::-1].plot(kind="barh", figsize=(7, 7), color="#4C72B0")
ax.set_title("Top 25 features by gain (last fold)")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md("## 7. Submission")

code(r"""
submission = pd.DataFrame({"id": test_ids, "target": test_pred})
out_path = os.path.join(OUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print("Wrote", out_path, submission.shape)
submission.head()
""")

md(r"""
## 8. Summary

- **Imbalance handled by the metric, not resampling.** Gini / AUC are rank-based
  and robust to the 27:1 class ratio; Stratified K-Fold keeps the positive rate
  stable across folds. `scale_pos_weight` is available but off by default because
  it changes calibration, not ranking.
- **Feature engineering** added `num_missing`, `mix_ind`, count encodings, and
  one-hot categoricals, while dropping the non-informative `ps_calc_*` block and
  rare/redundant indicators — all assembled into a memory-efficient sparse matrix.
- **LightGBM + Bayesian optimization** tuned over the suggested ranges, with a
  low final learning rate, early stopping, and fold-averaged test probabilities.
- The out-of-fold Gini is an unbiased estimate of leaderboard performance; for
  reference, strong public solutions on this competition score ~0.29 Gini.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

with open("HW4_Porto_Seguro.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"Wrote HW4_Porto_Seguro.ipynb with {len(cells)} cells")
