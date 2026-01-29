import numpy as np

# -----------------------------
# Preprocessing utilities
# -----------------------------
def fit_preprocess(train_df, feature_plan):
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


def transform(df, feature_plan, params):
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
                    onehot[i, cat_to_idx[v]] = 1.0
            X_parts.append(onehot)

    return np.hstack(X_parts)  # (n, d)


def add_intercept(X_row_major):
    # X: (n, d) -> (n, d+1)
    return np.hstack([np.ones((X_row_major.shape[0], 1)), X_row_major])


def to_column_major(X_row_major, y_vec):
    # SGD expects X: (d, n), y: (1, n)
    X = X_row_major.T
    y = y_vec.reshape(1, -1)
    return X, y


# -----------------------------
# SGD
# -----------------------------
def sgd(X, y, J, dJ, w0, step_size_fn, max_iter, seed=42):
    rng = np.random.default_rng(seed)
    w = w0
    fs = []
    ws = [w.copy()]

    n = X.shape[1]
    for i in range(max_iter):
        idx = int(rng.integers(0, n))
        Xi = X[:, idx:idx+1]  # (d,1)
        yi = y[:, idx:idx+1]  # (1,1)

        cost = J(Xi, yi, w)
        grad = dJ(Xi, yi, w)

        step = step_size_fn(i)
        w = w - step * grad

        fs.append(float(cost))
        ws.append(w.copy())

    return w, fs, ws


# -----------------------------
# Ridge: J and dJ (single-sample)
# J = 1/2 (x^T w - y)^2 + (lam/2)||w||^2
# (do not regularize intercept w[0])
# -----------------------------
def make_ridge_J_dJ(lam):
    # Implement the following functions
    def J(Xi, yi, w): # the function will change
        return 0.0 

    def dJ(Xi, yi, w): # the function will change
        return np.zeros_like(w)

    return J, dJ

# -----------------------------
# K-fold indices
# -----------------------------
def kfold_indices(n, k=10, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, k)
