from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import Adam
from torchdiffeq import odeint as odeint_torch
from ..models.pytorch import NeuralODE

@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 800
    widths: tuple = (32, 32)
    act: str = "tanh"
    time_invariant: bool = True
    reg_lambda: float = 0.0
    rtol: float = 1e-3
    atol: float = 1e-4
    method: str = "dopri5"

def mse(a, b): 
    return ((a - b) ** 2).mean()

def l2_reg(model: nn.Module) -> torch.Tensor:
    return sum((p.pow(2).sum() for p in model.parameters()))

def predict(model: NeuralODE, y0, t, rtol, atol, method):
    return odeint_torch(model.rhs, y0, t, rtol=rtol, atol=atol, method=method)

def train_one_seed(t, y_noisy, y0, cfg: TrainConfig, device="cpu"):
    t = t.to(device)
    y_noisy = y_noisy.to(device)
    y0 = y0.to(device)

    model = NeuralODE(cfg.widths, cfg.act, cfg.time_invariant).to(device)
    opt = Adam(model.parameters(), lr=cfg.lr)

    for _ in range(cfg.epochs):
        y_pred = predict(model, y0, t, cfg.rtol, cfg.atol, cfg.method)
        loss = mse(y_pred, y_noisy)
        if cfg.reg_lambda > 0:
            loss = loss + cfg.reg_lambda * l2_reg(model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        final_pred = predict(model, y0, t, cfg.rtol, cfg.atol, cfg.method)
        final_mse = mse(final_pred, y_noisy).item()

    return {"mse": final_mse, "model": model}
