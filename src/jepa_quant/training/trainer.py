"""Phased JEPA training loop.

Assembles the components, builds two optimizer param groups (projections vs
adapters), and runs the schedule from CLAUDE.md:

* Phase 1 (steps < ``warmup_proj_steps``): projection layers (and regularizer
  params, e.g. codebook) only. Encoder backbone + predictor adapters frozen.
* Phase 2 (steps >= ``warmup_proj_steps``): the predictor adapters (LoRA /
  transformer body) join at ``lr_adapter``; the price-encoder backbone joins at
  its own ``lr_encoder`` (higher — the ``transformer`` backend trains from
  scratch, so the LoRA rate barely moves it). Each Phase-2 group ramps in via a
  warmup -> cosine lr schedule for stability.

After every optimizer step the EMA target encoder is updated (never via the
optimizer). The total loss is ``L_jepa + regularizer_loss``; regularization is
applied to ``z_price`` only.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import JEPAConfig
from ..encoders.price_encoder import build_price_encoder
from ..encoders.target_encoder import TargetEncoder
from ..encoders.text_encoder import build_text_encoder
from ..predictor import build_predictor
from ..regularization.base import Regularizer, build_regularizer
from .jepa_loss import JepaKind, jepa_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device string to an actually-available device.

    ``"cpu"`` is always honoured. A requested accelerator is used when present;
    otherwise we fall back to any other available accelerator (CUDA or Apple
    MPS) before finally settling on CPU. This stops Apple-silicon machines from
    silently running on CPU just because the default request was ``"cuda"``.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and _mps_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class JEPAComponents:
    price_encoder: nn.Module
    target_encoder: TargetEncoder
    predictor: nn.Module
    regularizer: Regularizer
    text_encoder: Optional[nn.Module]


def build_components(cfg: JEPAConfig) -> JEPAComponents:
    price_encoder = build_price_encoder(cfg.price_encoder)
    text_encoder = build_text_encoder(cfg.text_encoder)
    predictor = build_predictor(
        cfg.predictor,
        in_dim=cfg.price_encoder.latent_dim,
        use_text=text_encoder is not None,
    )
    regularizer = build_regularizer(cfg)
    # Target encoder is an EMA copy of the *initial* price encoder.
    target_encoder = TargetEncoder(price_encoder, cfg.ema.decay)
    return JEPAComponents(price_encoder, target_encoder, predictor, regularizer, text_encoder)


def _infinite(loader: Iterable) -> Iterator:
    while True:
        yield from loader


class JEPATrainer:
    def __init__(
        self,
        cfg: JEPAConfig,
        components: JEPAComponents,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        jepa_kind: JepaKind = "cosine",
    ) -> None:
        # An empty train loader would make ``_infinite`` spin forever yielding
        # nothing (``next`` never returns) — fail loudly instead of hanging.
        # ``drop_last=True`` empties the loader whenever the dataset has fewer
        # than ``batch_size`` windows, so this is a realistic small-data trap.
        if len(train_loader) == 0:
            raise ValueError(
                f"train_loader is empty ({len(train_loader.dataset)} windows, "
                f"batch_size={cfg.train.batch_size}, drop_last=True). Reduce "
                "batch_size or provide more data."
            )

        self.cfg = cfg
        self.c = components
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.jepa_kind = jepa_kind
        self.device = resolve_device(cfg.train.device)

        set_seed(cfg.train.seed)
        self._to_device()
        self._build_optimizer()
        self.phase = 1
        self.history: list[dict] = []

    # ---- setup -----------------------------------------------------------

    def _to_device(self) -> None:
        self.c.price_encoder.to(self.device)
        self.c.target_encoder.to(self.device)
        self.c.predictor.to(self.device)
        self.c.regularizer.to(self.device)
        if self.c.text_encoder is not None:
            self.c.text_encoder.to(self.device)

    def _proj_params(self) -> list[nn.Parameter]:
        params = list(self.c.price_encoder.projection_parameters())
        params += list(self.c.predictor.projection_parameters())
        if self.c.text_encoder is not None:
            params += list(self.c.text_encoder.projection_parameters())
        params += list(self.c.regularizer.parameters())  # e.g. codebook codes
        return params

    def _encoder_params(self) -> list[nn.Parameter]:
        """Price- (and text-) encoder backbone params unfrozen in Phase 2. On
        the ``transformer`` backend these are trained from scratch, so they get
        their own higher lr (``lr_encoder``) rather than the LoRA rate."""
        params = list(self.c.price_encoder.tunable_backbone_parameters())
        if self.c.text_encoder is not None:
            params += list(self.c.text_encoder.tunable_backbone_parameters())
        return params

    def _predictor_adapter_params(self) -> list[nn.Parameter]:
        """Predictor adapters (LoRA / transformer body) enabled in Phase 2 at
        ``lr_adapter``."""
        return list(self.c.predictor.adapter_parameters())

    def _build_optimizer(self) -> None:
        tr = self.cfg.train
        encoder_params = self._encoder_params()
        predictor_params = self._predictor_adapter_params()
        # Everything that joins in Phase 2 starts frozen (re-enabled in
        # ``_enter_phase_2``). Held together only for the freeze/unfreeze toggle.
        self.phase2_params = encoder_params + predictor_params
        for p in self.phase2_params:
            p.requires_grad_(False)

        # One param group per lr, plus the step each becomes active. The encoder
        # backbone gets its own (higher) ``lr_encoder`` separate from the
        # predictor adapters, so a from-scratch backbone actually moves while
        # LoRA stays at its gentle ``lr_adapter``. ``_proj_params`` train from
        # step 1; the Phase-2 groups switch on at ``warmup_proj_steps``.
        p2 = tr.warmup_proj_steps
        groups: list[dict] = [{"params": self._proj_params(), "lr": tr.lr_proj}]
        activations: list[int] = [0]
        self._group_names: list[str] = ["proj"]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": tr.lr_encoder})
            activations.append(p2)
            self._group_names.append("enc")
        if predictor_params:
            groups.append({"params": predictor_params, "lr": tr.lr_adapter})
            activations.append(p2)
            self._group_names.append("ada")
        self.opt = torch.optim.AdamW(groups, weight_decay=tr.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lr_lambda=[self._lr_lambda(a) for a in activations]
        )

    def _lr_lambda(self, activation_step: int):
        """Per-group lr multiplier: 0 until the group activates, linear warmup
        over ``lr_warmup_steps``, then cosine decay to ``lr_min_factor`` x base
        by ``max_steps``. Guards keep it finite when ``lr_warmup_steps`` exceeds
        ``max_steps`` (tiny test configs). ``LambdaLR`` passes the 0-based step."""
        tr = self.cfg.train
        warmup = max(1, tr.lr_warmup_steps)
        floor = tr.lr_min_factor
        decay_start = activation_step + warmup
        decay_span = max(1, tr.max_steps - decay_start)

        def fn(step: int) -> float:
            if step < activation_step:
                return 0.0
            local = step - activation_step
            if local < warmup:
                return (local + 1) / warmup
            progress = min(1.0, (step - decay_start) / decay_span)
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return fn

    def _enter_phase_2(self) -> None:
        for p in self.phase2_params:
            p.requires_grad_(True)
        self.phase = 2

    # ---- codebook init ---------------------------------------------------

    def maybe_kmeans_init(self, n_samples: int = 2048) -> None:
        reg = self.c.regularizer
        if getattr(reg, "_initialised", True):
            return
        self.c.price_encoder.eval()
        collected: list[torch.Tensor] = []
        total = 0
        with torch.no_grad():
            for batch in self.train_loader:
                z = self.c.price_encoder(batch["context"].to(self.device))
                collected.append(z.cpu())
                total += z.size(0)
                if total >= n_samples:
                    break
        reg.maybe_kmeans_init(torch.cat(collected, dim=0))
        self.c.price_encoder.train()

    # ---- loss ------------------------------------------------------------

    def _compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        context = batch["context"].to(self.device)
        target = batch["target"].to(self.device)
        z_price = self.c.price_encoder(context)
        z_used = self.c.regularizer.transform(z_price)
        z_pred = self.c.predictor(z_used, None)  # text path not yet wired to dataset
        with torch.no_grad():
            z_target = self.c.target_encoder(target)
        l_jepa = jepa_loss(z_pred, z_target, self.jepa_kind)
        l_reg, reg_metrics = self.c.regularizer.loss(z_price, z_used)
        loss = l_jepa + l_reg
        metrics = {"loss": float(loss.detach()), "jepa": float(l_jepa.detach()), **reg_metrics}
        return loss, metrics

    # ---- train -----------------------------------------------------------

    def _trainable(self) -> list[nn.Parameter]:
        return [p for g in self.opt.param_groups for p in g["params"] if p.requires_grad]

    def train(self) -> list[dict]:
        cfg = self.cfg.train
        self.maybe_kmeans_init()
        self.c.price_encoder.train()
        self.c.predictor.train()
        data_iter = _infinite(self.train_loader)
        accum = max(1, cfg.grad_accum_steps)

        for step in range(1, cfg.max_steps + 1):
            if self.phase == 1 and step > cfg.warmup_proj_steps:
                self._enter_phase_2()

            self.opt.zero_grad(set_to_none=True)
            agg: dict[str, float] = {}
            for _ in range(accum):
                batch = next(data_iter)
                loss, metrics = self._compute_loss(batch)
                (loss / accum).backward()
                for k, v in metrics.items():
                    agg[k] = agg.get(k, 0.0) + v / accum

            torch.nn.utils.clip_grad_norm_(self._trainable(), cfg.grad_clip)
            self.opt.step()
            self.c.target_encoder.update(self.c.price_encoder)  # post-step EMA

            if step % cfg.log_every == 0 or step == 1:
                lrs = {
                    f"lr_{n}": g["lr"]
                    for n, g in zip(self._group_names, self.opt.param_groups)
                }
                record = {"step": step, "phase": self.phase, **agg, **lrs}
                self.history.append(record)
                reg_str = " ".join(
                    f"{k.split('/')[-1]}={agg[k]:.4f}" for k in agg if k.startswith("reg/")
                )
                lr_str = " ".join(f"{k}={v:.2e}" for k, v in lrs.items())
                print(
                    f"[{step:>5}/{cfg.max_steps}] phase={self.phase} "
                    f"loss={agg['loss']:.4f} jepa={agg['jepa']:.4f} {reg_str} {lr_str}"
                )

            if self.val_loader is not None and step % cfg.val_every == 0:
                val = self.validate()
                if self.history:
                    self.history[-1]["val_jepa"] = val["val_jepa"]
                print(f"        val_jepa={val['val_jepa']:.4f} z_std={val['val_z_std']:.4f}")

            # Advance the lr schedule last, so the lr logged above is the one
            # actually used for this step (LambdaLR set step-1's value on entry).
            self.sched.step()

        return self.history

    @torch.no_grad()
    def validate(self, max_batches: int = 20) -> dict:
        self.c.price_encoder.eval()
        self.c.predictor.eval()
        jepa_vals: list[float] = []
        z_stds: list[float] = []
        for i, batch in enumerate(self.val_loader or []):
            context = batch["context"].to(self.device)
            target = batch["target"].to(self.device)
            z_price = self.c.price_encoder(context)
            z_used = self.c.regularizer.transform(z_price)
            z_pred = self.c.predictor(z_used, None)
            z_target = self.c.target_encoder(target)
            jepa_vals.append(float(jepa_loss(z_pred, z_target, self.jepa_kind)))
            z_stds.append(float(z_price.std(dim=0).mean()))
            if i + 1 >= max_batches:
                break
        self.c.price_encoder.train()
        self.c.predictor.train()
        return {
            "val_jepa": float(np.mean(jepa_vals)) if jepa_vals else float("nan"),
            "val_z_std": float(np.mean(z_stds)) if z_stds else float("nan"),
        }
