"""Windowed price dataset for JEPA training.

Each parquet in ``data_dir`` (one per ticker; columns
``open/high/low/close/vwap/volume`` indexed by timestamp ``ts``) is turned into
per-timestep log-return features, then sliced into overlapping samples:

* ``context`` — ``[L, F]`` window fed to the (context) price encoder.
* ``target``  — ``[H, F]`` future window fed to the EMA target encoder.

Normalization is causal: per-sample mean/std are computed on the context window
only and applied to both windows, so no future information leaks. The
train/val split is chronological *per ticker* (earliest windows train, latest
validate).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import DataConfig, JEPAConfig

Split = Literal["train", "val"]


def _timestep_features(df: pd.DataFrame, price_cols: Sequence[str]) -> np.ndarray:
    """[T-1, F] log-return / log-volume-change features (first row dropped)."""
    cols = []
    for col in price_cols:
        s = df[col].astype("float64").to_numpy()
        if col == "volume":
            logs = np.log1p(np.clip(s, 0.0, None))
        else:
            logs = np.log(np.clip(s, 1e-8, None))
        cols.append(np.diff(logs, prepend=logs[0]))
    feats = np.stack(cols, axis=1).astype("float32")  # [T, F]
    return feats[1:]  # drop the artificial zero-return first row


class PriceWindowDataset(Dataset):
    def __init__(self, cfg: DataConfig, split: Split) -> None:
        self.cfg = cfg
        self.split = split
        self.span = cfg.context_length + cfg.horizon
        self._feats: list[np.ndarray] = []
        self._index: list[Tuple[int, int]] = []  # (ticker_idx, start)

        files = sorted(Path(cfg.data_dir).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {cfg.data_dir!r}")

        for path in files:
            df = pd.read_parquet(path).sort_index()
            feats = _timestep_features(df, cfg.price_cols)
            if len(feats) < self.span:
                continue
            ti = len(self._feats)
            self._feats.append(feats)
            starts = list(range(0, len(feats) - self.span + 1, cfg.stride))
            cut = int(len(starts) * (1.0 - cfg.val_fraction))
            chosen = starts[:cut] if split == "train" else starts[cut:]
            self._index.extend((ti, s) for s in chosen)

        if not self._index:
            raise ValueError(
                f"No {split} windows produced — check context_length/horizon "
                f"vs series length ({self.span} needed)."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ti, start = self._index[i]
        L = self.cfg.context_length
        window = self._feats[ti][start : start + self.span]  # [L+H, F]
        context = window[:L]
        target = window[L:]
        if self.cfg.normalize:
            mu = context.mean(axis=0, keepdims=True)
            sigma = context.std(axis=0, keepdims=True) + 1e-6
            context = (context - mu) / sigma
            target = (target - mu) / sigma
        return {
            "context": torch.from_numpy(np.ascontiguousarray(context)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
        }


def build_dataloaders(cfg: JEPAConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = PriceWindowDataset(cfg.data, "train")
    val_ds = PriceWindowDataset(cfg.data, "val")
    common = dict(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, drop_last=True)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader
