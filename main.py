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
def load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type '{ext}'. Use .csv or .parquet")

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
            nthread=args.cores,
            eval_metric="logloss",
            verbosity=0
        )
    else:
        model = xgb.XGBRegressor(
            tree_method="hist",
            nthread=args.cores,
            eval_metric="rmse",
            verbosity=0
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

