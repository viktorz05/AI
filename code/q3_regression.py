import numpy as np
import pandas as pd

from q3_helper import (
    fit_preprocess, transform, add_intercept, to_column_major,
    sgd, make_ridge_J_dJ, kfold_indices
)

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "car-dataset.tsv"

feature_plan = {
    "cylinders": "standard",
    "displacement": "standard",
    "horsepower": "standard",
    "weight": "standard",
    "acceleration": "standard",
    "model_year": "standard",
    "origin": "one-hot",
    "car_name": "drop",
}

lambda_values = [0.0, 0.0, 0.0]   # test for 3 lambda values
MAX_ITER = 20000

def step_size_fn(i): # notice how I defined step size
    return 0.05 / np.sqrt(i + 1)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH, sep="\t")
y_all = df["mpg"].astype(float).to_numpy()

folds = kfold_indices(len(df), k=10, seed=42)

# -----------------------------
# Metrics
# -----------------------------
def rmse(y_true, y_pred):
    # Implement rmse metric
    return 1 # this line will change


def mae(y_true, y_pred):
    # Implement mae metric
    return 1 # this line will change

print("\nRidge Regression (SGD) — 10-fold CV\n")
print("lambda |  RMSE  |  MAE")
print("------------------------")

for lam in lambda_values:
    J, dJ = make_ridge_J_dJ(lam)
    rmses, maes = [], []

    for i in range(10):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(10) if j != i])

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        params = fit_preprocess(train_df, feature_plan)

        X_train = add_intercept(transform(train_df, feature_plan, params))
        X_test = add_intercept(transform(test_df, feature_plan, params))

        Xtr, ytr = to_column_major(X_train, y_train)
        w0 = np.zeros((Xtr.shape[0], 1), dtype=float)

        w, _, _ = sgd(Xtr, ytr, J, dJ, w0, step_size_fn, MAX_ITER, seed=42) # perform sgd

        y_pred = (X_test @ w).reshape(-1)

        rmses.append(rmse(y_test, y_pred))
        maes.append(mae(y_test, y_pred))

    print(f"{lam:<6} | {np.mean(rmses):.3f} | {np.mean(maes):.3f}")
