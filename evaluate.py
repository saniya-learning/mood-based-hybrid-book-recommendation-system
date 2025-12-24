import numpy as np
import pandas as pd
from .data_loader import load_ratings_safe

def train_test_split(ratings, test_ratio=0.2, min_items=5):
    grouped = ratings.groupby("User-ID")
    train_rows, test_rows = [], []

    for uid, group in grouped:
        if len(group) < min_items:
            train_rows.append(group)
            continue
        group = group.sample(frac=1.0, random_state=42)
        n_test = max(1, int(len(group) * test_ratio))
        test_rows.append(group.iloc[:n_test])
        train_rows.append(group.iloc[n_test:])

    train = pd.concat(train_rows, ignore_index=True) if train_rows else ratings.iloc[0:0]
    test = pd.concat(test_rows, ignore_index=True) if test_rows else ratings.iloc[0:0]
    return train, test

def global_mean_baseline(train, test):
    # predict every rating as the global mean rating in train
    mu = train["Rating"].mean()
    y_true = test["Rating"].values
    y_pred = np.full_like(y_true, fill_value=mu, dtype=float)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return mu, rmse, mae

def main():
    print("Loading ratings...")
    ratings = load_ratings_safe(sample_size=20000)
    print("Ratings shape:", ratings.shape)

    print("Splitting train/test...")
    train, test = train_test_split(ratings, test_ratio=0.2, min_items=5)
    print("Train:", train.shape, "Test:", test.shape)

    print("Global-mean baseline (rating prediction)...")
    mu, rmse, mae = global_mean_baseline(train, test)
    print(f"Global mean rating (train): {mu:.3f}")
    print(f"RMSE on test: {rmse:.3f}")
    print(f"MAE on test:  {mae:.3f}")

if __name__ == "__main__":
    main()
