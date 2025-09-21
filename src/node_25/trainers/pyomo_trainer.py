from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional, Dict
import numpy as np

from ..core.collocation import chebyshev_lobatto_nodes, chebyshev_diff_matrix
from ..models.pyomo_model import CollocConfig, NeuralODEPyomo

@dataclass
class PyomoTrainConfig:
    widths: Sequence[int] = (32, 32)
    act: str = "tanh"
    lambda_reg: float = 0.0
    fix_y0: bool = True
    time_invariant: bool = True
    solver: str = "ipopt"
    solver_options: Optional[Dict[str, float]] = None
    N_nodes: Optional[int] = None  # default = len(t_uniform)

def train_collocation(t_uniform: np.ndarray, y_noisy_uniform: np.ndarray, y0: np.ndarray,
                      t0: float, t1: float, cfg: PyomoTrainConfig):
    N = cfg.N_nodes or len(t_uniform)
    t_nodes = chebyshev_lobatto_nodes(N, t0, t1)
    D = chebyshev_diff_matrix(N, t0, t1)

    # interpolate noisy observations to nodes
    y_obs = np.empty((N, y_noisy_uniform.shape[1]), dtype=np.float64)
    for d in range(y_noisy_uniform.shape[1]):
        y_obs[:, d] = np.interp(t_nodes, t_uniform, y_noisy_uniform[:, d]).astype(np.float64)

    model_cfg = CollocConfig(widths=tuple(cfg.widths), act=cfg.act,
                             lambda_reg=cfg.lambda_reg, fix_y0=cfg.fix_y0,
                             time_invariant=cfg.time_invariant)
    trainer = NeuralODEPyomo(t_nodes, y_obs, D, y0.astype(np.float64), model_cfg)
    trainer.build()
    trainer.solve(tee=True, solver=cfg.solver, options=cfg.solver_options or {"max_iter": 3000})
    out = trainer.extract()

    # interpolate prediction back to uniform grid for reporting
    y_pred_uniform = np.empty_like(y_noisy_uniform, dtype=np.float64)
    for d in range(y_pred_uniform.shape[1]):
        y_pred_uniform[:, d] = np.interp(t_uniform, out["t"], out["Y"][:, d])

    return {
        "t_nodes": out["t"],
        "y_pred_nodes": out["Y"],
        "t_uniform": t_uniform,
        "y_pred_uniform": y_pred_uniform,
        "weights": out["weights"],
        "biases": out["biases"],
    }
