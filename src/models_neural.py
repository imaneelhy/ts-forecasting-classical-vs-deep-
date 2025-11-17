# src/models_neural.py
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out.squeeze(-1)


def _make_loader(X, y, batch_size=32, shuffle=True):
    X_t = torch.from_numpy(X).float().unsqueeze(-1)  # (N, seq, 1)
    y_t = torch.from_numpy(y).float()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@dataclass
class LSTMConfig:
    hidden_size: int = 32
    num_layers: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 20


def train_lstm_and_forecast(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: LSTMConfig = LSTMConfig(),
) -> Dict[str, np.ndarray]:
    """
    Train an LSTM forecaster and return dictionary with test predictions (scaled).
    """
    train_loader = _make_loader(X_train, y_train, batch_size=config.batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch_size=config.batch_size, shuffle=False)
    test_loader = _make_loader(X_test, y_test, batch_size=config.batch_size, shuffle=False)

    model = LSTMForecaster(
        input_size=1,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * len(xb)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch}: train MSE={train_loss:.4f}, val MSE={val_loss:.4f}")

    # Forecast on test
    model.eval()
    preds_test = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            preds = model(xb)
            preds_test.append(preds.cpu().numpy())
    preds_test = np.concatenate(preds_test, axis=0)

    return {"forecast_scaled": preds_test}
