# src/models_classical.py
from typing import Tuple

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def naive_forecast(history: np.ndarray) -> float:
    """Naïve forecast: last observed value."""
    return history[-1]


def moving_average_forecast(history: np.ndarray, window: int) -> float:
    """Moving average of last `window` values."""
    window = min(window, len(history))
    return history[-window:].mean()


def baselines_forecasts(
    test_scaled: np.ndarray,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute naïve and moving-average one-step-ahead forecasts on the test segment.
    Returns predictions aligned with y_test (length len(test_scaled) - window_size).
    """
    naive_preds = []
    ma_preds = []

    for i in range(window_size, len(test_scaled)):
        history = test_scaled[i - window_size : i]
        naive_preds.append(naive_forecast(history))
        ma_preds.append(moving_average_forecast(history, window=window_size))

    return np.array(naive_preds), np.array(ma_preds)


def arima_rolling_forecast(
    train_scaled: np.ndarray,
    val_scaled: np.ndarray,
    test_scaled: np.ndarray,
    window_size: int,
    order: Tuple[int, int, int] = (5, 1, 0),
) -> np.ndarray:
    """
    Rolling 1-step-ahead ARIMA forecast on scaled data.
    Returns predictions aligned with y_test (len(test_scaled) - window_size).
    """
    series_tv = np.concatenate([train_scaled, val_scaled])
    history = list(series_tv)
    forecasts = []

    for t in range(len(test_scaled)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps=1)[0]
        forecasts.append(yhat)
        history.append(test_scaled[t])

    forecasts = np.array(forecasts)
    # align with y_test (which corresponds to test_scaled[window_size:])
    return forecasts[window_size:]
