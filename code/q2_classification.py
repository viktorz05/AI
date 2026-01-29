import numpy as np
import pandas as pd

# -----------------------------
# 1) Load your TSV dataset
# -----------------------------
DATA_PATH = "car-dataset.tsv"  # You need to specify the path to the dataset (default: in current directory)
df = pd.read_csv(DATA_PATH, sep="\t")

# -----------------------------
# 2) Create binary labels
# -----------------------------
MPG_THRESHOLD = 0.0   # <-- put your Q1.a threshold here
y = (df["mpg"].astype(float) >= MPG_THRESHOLD).astype(int).to_numpy()

# -----------------------------
# 3) Preprocessing
# -----------------------------
feature_plan = { # change the features preporcessing based on your answer to Q1.b
    "cylinders": "standard",
    "displacement": "standard",
    "horsepower": "standard",
    "weight": "standard",
    "acceleration": "standard",
    "model_year": "standard",
    "origin": "one-hot",
    "car_name": "drop",
}

def fit_preprocess(train_df):
    params = {"standard": {}, "onehot": {}}

    for col, how in feature_plan.items():
        if how == "standard":
            mu = train_df[col].mean()
            sigma = train_df[col].std(ddof=0)
            if sigma == 0:
                sigma = 1.0
            params["standard"][col] = (mu, sigma)

        elif how == "one-hot":
            params["onehot"][col] = sorted(train_df[col].unique())

    return params

def transform(df, params):
    X_parts = []

    for col, how in feature_plan.items():
        if how == "drop":
            continue

        if how == "standard":
            mu, sigma = params["standard"][col]
            x = (df[col] - mu) / sigma
            X_parts.append(x.to_numpy().reshape(-1, 1))

        elif how == "one-hot":
            cats = params["onehot"][col]
            onehot = np.zeros((len(df), len(cats)))
            cat_to_idx = {c: i for i, c in enumerate(cats)}
            for i, v in enumerate(df[col]):
                if v in cat_to_idx:
                    onehot[i, cat_to_idx[v]] = 1
            X_parts.append(onehot)

    return np.hstack(X_parts)

# -----------------------------
# 4) KNN
# -----------------------------
def knn_predict(X_train, y_train, X_test, k):
    preds = []
    for x in X_test:
        # Implement KNN Prediction
        preds = [] # this line will change
    return np.array(preds)

# -----------------------------
# 5) Metrics
# -----------------------------
def accuracy(y_true, y_pred):
    return 1 # Implement accuracy metric - this line will change

def f1(y_true, y_pred):
    # Implement f1 metric
    return 1 # this line will change

# -----------------------------
# 6) 10-fold CV
# -----------------------------
def kfold_indices(n, k=10, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, k)

folds = kfold_indices(len(df), 10)

k_values = [1, 1, 1]   # test 3 K values - this line will change

print("\nKNN Classification Results (10-fold CV):\n")
print("K | Accuracy | F1-score")
print("------------------------")

for k in k_values:
    accs, f1s = [], []

    for i in range(10):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(10) if j != i])

        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]

        y_train = y[train_idx]
        y_test  = y[test_idx]

        params = fit_preprocess(train_df)
        X_train = transform(train_df, params)
        X_test  = transform(test_df, params)

        y_pred = knn_predict(X_train, y_train, X_test, k)

        accs.append(accuracy(y_test, y_pred))
        f1s.append(f1(y_test, y_pred))

    print(f"{k:2d} |   {np.mean(accs):.3f}   |   {np.mean(f1s):.3f}")
