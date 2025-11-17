# src/prepare_data.py
import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "daily-min-temperatures.csv"
)


def load_temperature_dataset(path: str = "./data/daily-min-temperatures.csv") -> pd.DataFrame:
    """
    Download (if needed) and load the daily minimum temperature dataset.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading dataset to {path} ...")
        df = pd.read_csv(DATA_URL, parse_dates=["Date"])
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path, parse_dates=["Date"])

    df = df.sort_values("Date")
    return df


def make_supervised(series: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a 1D series (scaled), create supervised pairs:
    X[i] = series[i-window_size:i], y[i] = series[i]
    """
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size : i])
        y.append(series[i])
    return np.array(X), np.array(y)


def prepare_data(window_size: int = 14) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Load data, split into train/val/test, scale (fit scaler on train only),
    and create supervised data for neural networks.

    Returns:
        df: original dataframe
        series_data: dict with scaled train/val/test and scaler
        nn_data: dict with X/y splits for NN and scaler
    """
    df = load_temperature_dataset()
    values = df["Temp"].values.astype("float32")
    N = len(values)

    train_end = int(N * 0.7)
    val_end = int(N * 0.85)

    train_raw = values[:train_end].reshape(-1, 1)
    val_raw = values[train_end:val_end].reshape(-1, 1)
    test_raw = values[val_end:].reshape(-1, 1)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw).astype("float32").flatten()
    val_scaled = scaler.transform(val_raw).astype("float32").flatten()
    test_scaled = scaler.transform(test_raw).astype("float32").flatten()

    X_train, y_train = make_supervised(train_scaled, window_size)
    X_val, y_val = make_supervised(val_scaled, window_size)
    X_test, y_test = make_supervised(test_scaled, window_size)

    nn_data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "window_size": window_size,
        "scaler": scaler,
    }

    series_data = {
        "train_scaled": train_scaled,
        "val_scaled": val_scaled,
        "test_scaled": test_scaled,
        "scaler": scaler,
    }

    return df, series_data, nn_data
