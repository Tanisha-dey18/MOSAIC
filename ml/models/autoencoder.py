"""PyTorch fully-connected autoencoder for window-level feature vectors.

Input: normalised feature vector z (d ~ 14). Architecture d -> 16 -> 4 -> 16 -> d
with ReLU activations and dropout. Trained on "normal" (calibration) windows
only, using mean-squared reconstruction loss

    L(z) = (1/d) * sum_j (z_j - zhat_j)^2

The anomaly score of a window is its reconstruction error; per-feature squared
errors are reported as contributions. Early stopping monitors the loss on a
participant-disjoint validation split; the best weights are checkpointed.

A temporal (LSTM/GRU) model was deliberately *not* implemented: each
participant has only ~5 min of calibration data (~30 overlapping windows), so a
sequence model would have too few independent training sequences to be
justified. See docs/report.md (Methodology) for this decision.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class _AENet(nn.Module):
    def __init__(self, d: int, hidden: int = 16, bottleneck: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    stopped_early: bool

    def to_dict(self) -> dict:
        return {"train_loss": self.train_loss, "val_loss": self.val_loss,
                "best_epoch": self.best_epoch, "stopped_early": self.stopped_early}


class AutoencoderScorer:
    def __init__(self, hidden: int = 16, bottleneck: int = 4, dropout: float = 0.1, lr: float = 1e-3,
                 weight_decay: float = 1e-4, epochs: int = 200, batch_size: int = 64, patience: int = 15,
                 seed: int = 42):
        self.params = dict(hidden=hidden, bottleneck=bottleneck, dropout=dropout, lr=lr, weight_decay=weight_decay,
                           epochs=epochs, batch_size=batch_size, patience=patience)
        self.seed = seed
        self.net: _AENet | None = None
        self.d: int | None = None
        self.history: TrainingHistory | None = None

    def fit(self, Z_train: np.ndarray, Z_val: np.ndarray | None = None) -> "AutoencoderScorer":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        p = self.params
        self.d = Z_train.shape[1]
        self.net = _AENet(self.d, p["hidden"], p["bottleneck"], p["dropout"])
        opt = torch.optim.Adam(self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        Xt = torch.tensor(Z_train, dtype=torch.float32)
        Xv = torch.tensor(Z_val if Z_val is not None and len(Z_val) else Z_train, dtype=torch.float32)
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, best_epoch, bad = np.inf, None, 0, 0
        tl, vl = [], []
        for epoch in range(p["epochs"]):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=gen)
            total = 0.0
            for i in range(0, len(Xt), p["batch_size"]):
                xb = Xt[perm[i:i + p["batch_size"]]]
                opt.zero_grad()
                loss = ((self.net(xb) - xb) ** 2).mean()
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            tl.append(total / len(Xt))
            self.net.eval()
            with torch.no_grad():
                v = float(((self.net(Xv) - Xv) ** 2).mean())
            vl.append(v)
            if v < best - 1e-6:
                best, best_epoch, bad = v, epoch, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}  # in-memory checkpoint
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.history = TrainingHistory(tl, vl, best_epoch, stopped_early=len(tl) < p["epochs"])
        return self

    def _recon_error(self, Z: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("autoencoder is not trained")
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(Z, dtype=torch.float32)
            return ((self.net(x) - x) ** 2).numpy()

    def score(self, Z: np.ndarray) -> np.ndarray:
        return self._recon_error(Z).mean(axis=1)

    def contributions(self, Z: np.ndarray) -> np.ndarray:
        return self._recon_error(Z) / Z.shape[1]

    # --- serialisation -------------------------------------------------
    def state_bytes(self) -> bytes:
        buf = io.BytesIO()
        torch.save({"state_dict": self.net.state_dict(), "d": self.d, "params": self.params, "seed": self.seed}, buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "AutoencoderScorer":
        blob = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        params = dict(blob["params"])
        obj = cls(seed=blob["seed"], **params)
        obj.d = blob["d"]
        obj.net = _AENet(obj.d, params["hidden"], params["bottleneck"], params["dropout"])
        obj.net.load_state_dict(blob["state_dict"])
        obj.net.eval()
        return obj
