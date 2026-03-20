#!/usr/bin/env python3
import os
import argparse
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import functools
print = functools.partial(print, flush=True)
import polars as pl

def load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext in (".parquet", ".pq"):
        # Read all columns (or select below)
        df = pl.read_parquet(path)  # multi-threaded
    elif ext == ".csv":
        df = pl.read_csv(path)      # multi-threaded
    else:
        raise ValueError("Unsupported file type")


    return df.to_pandas()

def preprocess_for_xgb(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # 1) Datetime columns -> epoch seconds (float32), without Series.view
    dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for c in dt_cols:
        s = pd.to_datetime(X[c], errors="coerce")
        # datetime -> int64 nanoseconds -> seconds
        X[c] = (s.astype("int64") // 10**9).astype("float32")

    # 2) Object columns -> category (keep as category for enable_categorical)
    obj_cols = X.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        X[c] = X[c].astype("category")

    # 3) Clean categorical columns that can break XGBoost:
    #    - all missing in this split
    #    - zero categories
    cat_cols = X.select_dtypes(include=["category"]).columns
    drop_cols = []
    for c in cat_cols:
        # If all values are NA -> XGBoost categorical path can explode
        if X[c].isna().all():
            drop_cols.append(c)
            continue

        # Ensure categories exist (rare but can happen)
        if len(X[c].cat.categories) == 0:
            drop_cols.append(c)
            continue

        # Remove unused categories (good hygiene)
        X[c] = X[c].cat.remove_unused_categories()

        # After removing unused, still empty?
        if len(X[c].cat.categories) == 0:
            drop_cols.append(c)

    if drop_cols:
        X = X.drop(columns=drop_cols)

    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV or Parquet dataset")
    parser.add_argument("--cores", type=int, required=True, help="Number of CPU cores")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--val_split", type=float, required=True, help="Validation split fraction (e.g. 0.2)")
    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...", flush=True)
    df = load_dataframe(args.data)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset")



    X = df.drop(columns=[args.target])
    X = preprocess_for_xgb(X)
    y = df[args.target]

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=42
    )

    # Detect task type (simple heuristic)
    if y.dtype == "object" or pd.api.types.is_bool_dtype(y) or y.nunique(dropna=True) < 20:
        task = "classification"
    else:
        task = "regression"

    print(f"Detected task: {task}", flush=True)
    print(f"Training with {args.cores} cores", flush=True)

    # Configure model
    if task == "classification":
        model = xgb.XGBClassifier(
            tree_method="hist",
            n_jobs=args.cores,
            n_estimators=200,
            eval_metric="logloss",
            verbosity=0,
            enable_categorical=True
        )
    else:
        model = xgb.XGBRegressor(
            tree_method="hist",
            n_jobs=args.cores,
            n_estimators=200,
            eval_metric="rmse",
            verbosity=0,
            enable_categorical=True
        )

    # Time training only
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_time = end - start

    # Validation predictions
    y_pred = model.predict(X_val)

    if task == "classification":
        score = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {score:.6f}", flush=True)
    else:
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f"Validation RMSE: {rmse:.6f}", flush=True)

    print(f"Training Time (seconds): {train_time:.4f}", flush=True)


if __name__ == "__main__":
    main()

