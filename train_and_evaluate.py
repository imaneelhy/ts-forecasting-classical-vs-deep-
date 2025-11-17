# train_and_evaluate.py
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.prepare_data import prepare_data
from src.models_classical import baselines_forecasts, arima_rolling_forecast
from src.models_neural import train_lstm_and_forecast, LSTMConfig


RESULTS_DIR = "results"


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0


def evaluate_series(y_true, y_pred):
    return {
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mape(y_true, y_pred)),
    }


def inverse_scale(arr_scaled, scaler):
    arr_scaled = np.asarray(arr_scaled).reshape(-1, 1)
    return scaler.inverse_transform(arr_scaled).flatten()


def main(window_size: int = 14):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df, series_data, nn_data = prepare_data(window_size=window_size)

    train_scaled = series_data["train_scaled"]
    val_scaled = series_data["val_scaled"]
    test_scaled = series_data["test_scaled"]
    scaler = series_data["scaler"]

    X_train = nn_data["X_train"]
    y_train = nn_data["y_train"]
    X_val = nn_data["X_val"]
    y_val = nn_data["y_val"]
    X_test = nn_data["X_test"]
    y_test = nn_data["y_test"]

    # Ground truth in original units
    y_test_inv = inverse_scale(y_test, scaler)

    # ---- Baselines (scaled) ----
    naive_scaled, ma_scaled = baselines_forecasts(test_scaled, window_size)
    naive_inv = inverse_scale(naive_scaled, scaler)
    ma_inv = inverse_scale(ma_scaled, scaler)

    metrics_naive = evaluate_series(y_test_inv, naive_inv)
    metrics_ma = evaluate_series(y_test_inv, ma_inv)

    # ---- ARIMA (rolling one-step) ----
    arima_scaled = arima_rolling_forecast(
        train_scaled, val_scaled, test_scaled, window_size, order=(5, 1, 0)
    )
    arima_inv = inverse_scale(arima_scaled, scaler)
    metrics_arima = evaluate_series(y_test_inv, arima_inv)

    # ---- LSTM ----
    lstm_out = train_lstm_and_forecast(X_train, y_train, X_val, y_val, X_test, y_test,
                                       config=LSTMConfig())
    lstm_scaled = lstm_out["forecast_scaled"]
    lstm_inv = inverse_scale(lstm_scaled, scaler)
    metrics_lstm = evaluate_series(y_test_inv, lstm_inv)

    # ---- Collect metrics ----
    metrics = {
        "naive": metrics_naive,
        "moving_average": metrics_ma,
        "arima": metrics_arima,
        "lstm": metrics_lstm,
    }

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    # ---- Plot forecasts ----
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label="Ground truth", linewidth=2)
    plt.plot(naive_inv, label="Naive")
    plt.plot(ma_inv, label="Moving avg")
    plt.plot(arima_inv, label="ARIMA")
    plt.plot(lstm_inv, label="LSTM")

    plt.xlabel("Test time index")
    plt.ylabel("Temperature (°C)")
    plt.title("One-step-ahead forecasts on test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "example_forecasts.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main(window_size=14)
