"""The LSTM forecaster, behind the same Model interface as the baselines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from energycast.utils import get_logger

logger = get_logger(__name__)


class LSTMError(ValueError):
    """Raised when the forecaster is asked to work on shapes it cannot use."""


class _LSTMNetwork(nn.Module):
    """Reads a (sequence_length, n_features) window into prediction_horizon hours.

    Takes the last timestep's hidden state, which has seen the whole lookback,
    and maps it straight to the horizon with no activation on the head.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        prediction_horizon: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        directions = 2 if bidirectional else 1
        self.head = nn.Linear(hidden_size * directions, prediction_horizon)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(windows)
        return self.head(outputs[:, -1, :])


class LSTMForecaster:
    """Trains an LSTM on materialised windows and forecasts (n, prediction_horizon).

    Satisfies the Model protocol structurally: it takes np.ndarray windows of
    shape (n, sequence_length, n_features) where the baselines take a tabular
    frame, but returns the same (n, prediction_horizon) so milestone 7 can score
    them against each other. It forecasts in the units it is handed — the scaler
    inverse-transform stays outside, symmetric to the baselines.

    Early stopping needs validation. Pass the real split; without it the last
    validation_fraction of the (chronological) windows is held out.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        learning_rate: float,
        batch_size: int,
        max_epochs: int,
        early_stopping_patience: int,
        lr_scheduler_factor: float,
        lr_scheduler_patience: int,
        seed: int,
        validation_fraction: float = 0.1,
    ) -> None:
        self.name = "lstm"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.seed = seed
        self.validation_fraction = validation_fraction
        self.device = torch.device("cpu")
        self.network: _LSTMNetwork | None = None

    @classmethod
    def from_settings(cls, validation_fraction: float = 0.1) -> LSTMForecaster:
        from energycast.config import get_settings

        lstm = get_settings().model.lstm
        return cls(
            hidden_size=lstm.hidden_size,
            num_layers=lstm.num_layers,
            dropout=lstm.dropout,
            bidirectional=lstm.bidirectional,
            learning_rate=lstm.learning_rate,
            batch_size=lstm.batch_size,
            max_epochs=lstm.max_epochs,
            early_stopping_patience=lstm.early_stopping_patience,
            lr_scheduler_factor=lstm.lr_scheduler.factor,
            lr_scheduler_patience=lstm.lr_scheduler.patience,
            seed=lstm.seed,
            validation_fraction=validation_fraction,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epoch_callback: Callable[[int, float, float], None] | None = None,
    ) -> LSTMForecaster:
        if X.ndim != 3:
            raise LSTMError(
                f"Expected windows of shape (n, sequence_length, n_features), got {X.shape}."
            )

        torch.manual_seed(self.seed)
        if X_val is None or y_val is None:
            X, y, X_val, y_val = self._holdout_tail(X, y)

        train_loader = self._loader(X, y, shuffle=True)
        val_loader = self._loader(X_val, y_val, shuffle=False)

        self.network = _LSTMNetwork(
            input_size=X.shape[2],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            prediction_horizon=y.shape[1],
        ).to(self.device)

        optimiser = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, factor=self.lr_scheduler_factor, patience=self.lr_scheduler_patience
        )
        criterion = nn.MSELoss()

        best_val = float("inf")
        best_state = self.network.state_dict()
        epochs_without_improvement = 0
        epochs_run = 0

        for _ in range(self.max_epochs):
            epochs_run += 1
            train_loss = self._train_epoch(train_loader, optimiser, criterion)
            val_loss = self._evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            if epoch_callback is not None:
                epoch_callback(epochs_run, train_loss, val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_patience:
                break

        self.network.load_state_dict(best_state)
        logger.info(
            "fitted lstm",
            extra={
                "event": "lstm_fitted",
                "epochs": epochs_run,
                "best_val_loss": best_val,
                "train_windows": len(X),
                "val_windows": len(X_val),
                "n_features": X.shape[2],
            },
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.network is None:
            raise LSTMError("Call fit before predict.")
        if X.ndim != 3:
            raise LSTMError(
                f"Expected windows of shape (n, sequence_length, n_features), got {X.shape}."
            )

        self.network.eval()
        loader = DataLoader(
            TensorDataset(self._as_tensor(X)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        batches = []
        with torch.no_grad():
            for (windows,) in loader:
                batches.append(self.network(windows.to(self.device)).cpu().numpy())
        return np.concatenate(batches)

    def _holdout_tail(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cut = int(len(X) * (1.0 - self.validation_fraction))
        if cut < 1 or cut >= len(X):
            raise LSTMError(
                f"{len(X)} window(s) cannot be split at validation_fraction "
                f"{self.validation_fraction}; pass X_val/y_val explicitly."
            )
        return X[:cut], y[:cut], X[cut:], y[cut:]

    def _loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(self._as_tensor(X), self._as_tensor(y))
        generator = torch.Generator().manual_seed(self.seed) if shuffle else None
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, generator=generator)

    @staticmethod
    def _as_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))

    def _train_epoch(
        self, loader: DataLoader, optimiser: torch.optim.Optimizer, criterion: nn.Module
    ) -> float:
        self.network.train()
        total = 0.0
        count = 0
        for windows, targets in loader:
            windows = windows.to(self.device)
            targets = targets.to(self.device)
            optimiser.zero_grad()
            loss = criterion(self.network(windows), targets)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(windows)
            count += len(windows)
        return total / count

    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.network.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for windows, targets in loader:
                windows = windows.to(self.device)
                targets = targets.to(self.device)
                loss = criterion(self.network(windows), targets)
                total += loss.item() * len(windows)
                count += len(windows)
        return total / count
