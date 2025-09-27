from __future__ import annotations

import numpy as np
import torch

from node_25.viz.plotting import plot_predictions
from node_25.core.datasets import SynthConfig, generate_synth
from node_25.trainers.pytorch_trainer import TrainConfig, train_one_seed
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Literal
from torchdiffeq import odeint as odeint_torch
from pathlib import Path

from node_25.pipelines.base_experiment import ExperimentSequentialBase, ExperimentConfig
from node_25.pipelines.utils import *

def to_torch(
    t_np: np.ndarray, y_true_np: np.ndarray, y_noisy_np: np.ndarray, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.as_tensor(t_np, dtype=torch.float32, device=device)
    y_true = torch.as_tensor(y_true_np, dtype=torch.float32, device=device)
    y_noisy = torch.as_tensor(y_noisy_np, dtype=torch.float32, device=device)
    return t, y_true, y_noisy

def rollout_torch(
    model, y0_t: torch.Tensor, t_t: torch.Tensor, rtol: float, atol: float, method: str
) -> torch.Tensor:
    with torch.no_grad():
        y_pred = odeint_torch(model.rhs, y0_t, t_t, rtol=rtol, atol=atol, method=method)
    return y_pred

# ----------------------------- PyTorch experiment -----------------------------
@dataclass(frozen=True)
class TorchConfig:
    amp: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    grad_clip: Optional[float] = None

@dataclass(frozen=True)
class PytorchExperimentConfig:
    base: ExperimentConfig
    torch: TorchConfig = TorchConfig()


class PytorchExperiment(ExperimentSequentialBase):
    backend = "pytorch"

    def __init__(self, cfg: PytorchExperimentConfig):
        super().__init__(cfg)
        # prepared once (can be resampled per seed if requested)
        self.t_np: np.ndarray
        self.y_true_np: np.ndarray
        self.y_noisy_np: np.ndarray

    # ---------------- data prep ----------------

    def on_init(self) -> None:
        # Prepare base dataset once
        self._prepare_train_data(seed=None)

    def _prepare_train_data(self, seed: Optional[int]):
        if seed is not None:
            np.random.seed(seed)
        base = SynthConfig(
            system=self.cfg.system,
            N=self.cfg.N,
            t0=self.cfg.t0,
            t1=self.cfg.t1,
            y0=self.cfg.y0,
            noise=self.cfg.noise,
        )
        self.t_np, self.y_true_np, self.y_noisy_np, _ = generate_synth(base)

    def _prepare_test_data(self, last_true_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        duration = float(self.cfg.t1 - self.cfg.t0)
        t0_test = float(self.t_np[-1])
        t1_test = t0_test + duration
        cfg_test = SynthConfig(
            system=self.cfg.system,
            N=self.cfg.N,
            t0=t0_test,
            t1=t1_test,
            y0=tuple(last_true_train.tolist()),
            noise=self.cfg.noise,
        )
        t_test_np, y_true_test_np, y_noisy_test_np, _ = generate_synth(cfg_test)
        return t_test_np, y_true_test_np, y_noisy_test_np

    # ---------------- training per seed ----------------

    def _train_seed(self, seed: int) -> Dict:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # resample per seed if requested
        if self.cfg.resample_data:
            self._prepare_train_data(seed=seed)

        # tensors
        t, y_true, y_noisy = to_torch(self.t_np, self.y_true_np, self.y_noisy_np, self.cfg.device)
        y0_t = y_true[0]

        # train
        trainer_cfg = TrainConfig(
            lr=self.cfg.lr,
            epochs=self.cfg.epochs,
            widths=self.cfg.widths,
            time_invariant=self.cfg.time_invariant,
            reg_lambda=self.cfg.reg,
            rtol=self.cfg.rtol,
            atol=self.cfg.atol,
            method=self.cfg.method,
            pretrain_fracs=self.cfg.pretrain_fracs,
            pretrain_epochs=self.cfg.pretrain_epochs,
        )
        out = train_one_seed(t, y_noisy, y0_t, trainer_cfg, device=self.cfg.device)
        model = out["model"]

        # train rollout
        y_pred_train = rollout_torch(model, y0_t, t, self.cfg.rtol, self.cfg.atol, self.cfg.method)
        y_pred_train_np = y_pred_train.detach().cpu().numpy()

        mse_train_true = float(np.mean((self.y_true_np - y_pred_train_np) ** 2))
        mse_train_noisy = float(np.mean((self.y_noisy_np - y_pred_train_np) ** 2))

        # test data (start from last true train state)
        t_test_np, y_true_test_np, y_noisy_test_np = self._prepare_test_data(self.y_true_np[-1])
        t_test = torch.as_tensor(t_test_np, dtype=torch.float32, device=self.cfg.device)
        y0_test = y_true[-1]
        y_pred_test = rollout_torch(model, y0_test, t_test, self.cfg.rtol, self.cfg.atol, self.cfg.method)
        y_pred_test_np = y_pred_test.detach().cpu().numpy()

        mse_test_true = float(np.mean((y_true_test_np - y_pred_test_np) ** 2))
        mse_test_noisy = float(np.mean((y_noisy_test_np - y_pred_test_np) ** 2))

        # save per-seed artifacts
        seed_dir = self.run_root / f"seed{seed}"
        ensure_dir(seed_dir)

        if self.cfg.save_preds:
            np.savez(
                seed_dir / "predictions_train_seed.npz",
                t=self.t_np,
                y_true=self.y_true_np,
                y_noisy=self.y_noisy_np,
                y_pred=y_pred_train_np,
            )
            np.savez(
                seed_dir / "predictions_test_seed.npz",
                t=t_test_np,
                y_true=y_true_test_np,
                y_noisy=y_noisy_test_np,
                y_pred=y_pred_test_np,
            )

        # plots
        if self.cfg.save_plots and plot_predictions is not None:
            figs = (self.cfg.plots_dir or (seed_dir / "figs"))
            figs = Path(figs)
            ensure_dir(figs)
            plot_predictions(
                self.t_np, self.y_true_np, y_pred_train_np,
                y_noisy=self.y_noisy_np,
                title=f"{self.cfg.system.upper()} · PyTorch · seed{seed} · train",
                save=figs / "train.png", show=False,
            )
            plot_predictions(
                t_test_np, y_true_test_np, y_pred_test_np,
                y_noisy=y_noisy_test_np,
                title=f"{self.cfg.system.upper()} · PyTorch · seed{seed} · test",
                save=figs / "test.png", show=False,
            )

        # per-seed metrics.json
        per_seed_metrics = {
            "system": self.cfg.system,
            "backend": "pytorch",
            "seed": seed,
            "epochs": self.cfg.epochs,
            "noise": self.cfg.noise,
            "widths": self.cfg.widths,
            "reg_lambda": self.cfg.reg,
            "time_invariant": bool(self.cfg.time_invariant),
            "device": self.cfg.device,
            "rtol": self.cfg.rtol,
            "atol": self.cfg.atol,
            "method": self.cfg.method,
            "mse_train_true": mse_train_true,
            "mse_train_noisy": mse_train_noisy,
            "mse_test_true": mse_test_true,
            "mse_test_noisy": mse_test_noisy,
        }
        save_json(per_seed_metrics, seed_dir / "metrics.json")

        return per_seed_metrics
