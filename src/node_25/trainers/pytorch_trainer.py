from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torchdiffeq import odeint as odeint_torch
from ..models.pytorch_model import NeuralODE

@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 800                  # main-stage epochs on the FULL time grid
    widths: tuple = (32, 32)
    act: str = "tanh"
    time_invariant: bool = True
    reg_lambda: float = 0.0
    rtol: float = 1e-3
    atol: float = 1e-4
    method: str = "dopri5"
    # ---- pretraining schedule ----
    pretrain_fracs: Tuple[float, ...] = ()   # e.g., (0.1, 0.2)
    pretrain_epochs: Tuple[int, ...] = ()    # e.g., (100, 200)

def mse(a, b): 
    return ((a - b) ** 2).mean()

def l2_reg(model: nn.Module) -> torch.Tensor:
    return sum((p.pow(2).sum() for p in model.parameters()))

def predict(model: NeuralODE, y0, t, rtol, atol, method):
    # Ensure rtol/atol live on the right device/dtype (esp. for MPS)
    rtol_t = torch.as_tensor(rtol, dtype=t.dtype, device=t.device)
    atol_t = torch.as_tensor(atol, dtype=t.dtype, device=t.device)
    return odeint_torch(model.rhs, y0, t, rtol=rtol_t, atol=atol_t, method=method)

def _train_for_epochs(model, opt, y0, t, y_noisy, cfg: TrainConfig):
    """One stage of training on a (possibly truncated) time grid."""
    for _ in range(cfg_stage_epochs := getattr(cfg, "_stage_epochs", 0) or cfg.epochs):
        y_pred = predict(model, y0, t, cfg.rtol, cfg.atol, cfg.method)
        loss = mse(y_pred, y_noisy)
        if cfg.reg_lambda > 0:
            loss = loss + cfg.reg_lambda * l2_reg(model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

def train_one_seed(t, y_noisy, y0, cfg: TrainConfig, device="cpu"):
    t = t.to(device)
    y_noisy = y_noisy.to(device)
    y0 = y0.to(device)

    model = NeuralODE(cfg.widths, cfg.act, cfg.time_invariant).to(device)
    opt = Adam(model.parameters(), lr=cfg.lr)

    # ---------------------- PRETRAINING STAGES ---------------------------
    if cfg.pretrain_fracs:
        # If user didn't provide matching epochs, apportion evenly
        if not cfg.pretrain_epochs:
            per = max(1, cfg.epochs // len(cfg.pretrain_fracs))
            pretrain_epochs = tuple(per for _ in cfg.pretrain_fracs)
        else:
            pretrain_epochs = cfg.pretrain_epochs
        if len(pretrain_epochs) != len(cfg.pretrain_fracs):
            raise ValueError("pretrain_fracs and pretrain_epochs must have the same length.")

        n_total = t.shape[0]
        for frac, stage_ep in zip(cfg.pretrain_fracs, pretrain_epochs):
            k = max(2, int(round(float(frac) * n_total)))
            t_sub = t[:k]
            y_noisy_sub = y_noisy[:k]
            # set stage epochs temporarily
            cfg._stage_epochs = int(stage_ep)
            _train_for_epochs(model, opt, y0, t_sub, y_noisy_sub, cfg)
        # cleanup helper attr
        if hasattr(cfg, "_stage_epochs"):
            delattr(cfg, "_stage_epochs")

    # ----- MAIN STAGE on full grid (can be set to 0 to skip) -----
    if cfg.epochs > 0:
        _train_for_epochs(model, opt, y0, t, y_noisy, cfg)

    # Final metrics on full grid
    with torch.no_grad():
        final_pred = predict(model, y0, t, cfg.rtol, cfg.atol, cfg.method)
        final_mse = mse(final_pred, y_noisy).item()

    return {"mse": final_mse, "model": model}
