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
from typing import Literal, Mapping, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import DataConfig, JEPAConfig

Split = Literal["train", "val"]


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """float32 ndarray -> tensor without crossing the torch<->numpy C-API.

    Third trap in the numpy<2-for-Moirai stack: Colab's preinstalled torch
    wheel is built against NumPy 2, but uni2ts pins NumPy to 1.26. A
    numpy-2-built ``torch.from_numpy`` then refuses a numpy-1 array with
    ``expected np.ndarray (got numpy.ndarray)`` (the repr is identical; the
    C-level type identity differs). We can't cheaply downgrade the CUDA torch
    wheel, so we go through raw bytes instead: ``tobytes()`` is pure-NumPy and
    ``torch.frombuffer`` reads bytes with no NumPy involvement, so neither
    side's ABI is exercised. This is the only torch<->numpy crossing in the
    hot path — once samples are tensors, the rest of training is pure torch.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    # bytearray (writable) avoids torch's non-writable-buffer warning.
    flat = torch.frombuffer(bytearray(arr.tobytes()), dtype=torch.float32)
    return flat.reshape(arr.shape)


def _read_price_frame(path: Path) -> dict[str, np.ndarray]:
    """Read a price parquet to ``{column: ndarray}`` — pandas-free on purpose.

    Two version traps motivate going straight through Arrow to NumPy and never
    calling ``to_pandas``:

    1. The files store a tz-aware datetime ``ts`` index. Letting pandas rebuild
       it from the parquet metadata trips a pandas<->pyarrow bug (Colab,
       pandas 2.1.x + newer pyarrow): ``datetime64 values must have a unit
       specified``.
    2. When ``uni2ts`` pulls NumPy below 2.0 but Colab's pandas/pyarrow wheels
       were built against NumPy 2, the pyarrow->pandas bridge raises
       ``expected numpy.ndarray, got numpy.ndarray`` (a C-ABI mismatch).
       pyarrow->NumPy stays consistent, so we stop at NumPy.

    Rows are ordered on the raw ``ts`` epoch in Arrow, then ``ts`` is dropped —
    the dataset needs price columns in chronological order, never the
    timestamps. ``_timestep_features`` reads the returned mapping by column.
    """
    table = pq.read_table(path)
    if "ts" in table.column_names:
        ts = table.column("ts")
        # Cast tz-aware timestamp -> int64 epoch so ordering never touches the
        # fragile datetime->pandas conversion that raises the unit error.
        key = pc.cast(ts, pa.int64(), safe=False) if pa.types.is_timestamp(ts.type) else ts
        table = table.take(pc.sort_indices(key)).drop(["ts"])
    return {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}


def _timestep_features(
    df: Mapping[str, np.ndarray], price_cols: Sequence[str]
) -> np.ndarray:
    """[T-1, F] log-return / log-volume-change features (first row dropped).

    ``df`` is any column->array mapping (the pandas-free reader's dict, or a
    DataFrame); columns are coerced with ``np.asarray`` so both work.
    """
    cols = []
    for col in price_cols:
        s = np.asarray(df[col], dtype="float64")
        if col == "volume":
            logs = np.log1p(np.clip(s, 0.0, None))
        else:
            logs = np.log(np.clip(s, 1e-8, None))
        # Explicit first-difference instead of np.diff(prepend=...): the prepend
        # kwarg defaults to the np._NoValue sentinel, and a half-reloaded numpy
        # (autoreload + the uni2ts numpy<2 downgrade on Colab) leaves a stale
        # sentinel that leaks into the subtraction ("unsupported operand for -:
        # '_NoValueType'"). first row = 0 (dropped below), same as prepend=logs[0].
        diff = np.empty_like(logs)
        diff[0] = 0.0
        diff[1:] = logs[1:] - logs[:-1]
        cols.append(diff)
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
            df = _read_price_frame(path)
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
            "context": _to_tensor(context),
            "target": _to_tensor(target),
        }


def build_dataloaders(cfg: JEPAConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = PriceWindowDataset(cfg.data, "train")
    val_ds = PriceWindowDataset(cfg.data, "val")
    common = dict(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    # Train drops the last partial batch (stable batch stats for VICReg); val
    # keeps it so a small validation split is not silently emptied.
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader
