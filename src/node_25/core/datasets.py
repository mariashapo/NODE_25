from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from .ode_problems import RHS_REGISTRY

Array = np.ndarray

@dataclass
class SynthConfig:
    system: str = "ho"                 # "ho" | "vdp" | "do"
    N: int = 200
    t0: float = 0.0
    t1: float = 10.0
    y0: Tuple[float, float] = (0.0, 1.0)
    noise: float = 0.1                 # Gaussian std
    params: Optional[Dict] = None
    seed: Optional[int] = None
    rtol: float = 1e-7
    atol: float = 1e-9
    method: str = "RK45"

def generate_synth(cfg: SynthConfig):
    rng = np.random.default_rng(cfg.seed)
    t_eval = np.linspace(cfg.t0, cfg.t1, cfg.N, dtype=np.float32)
    y0 = np.array(cfg.y0, dtype=np.float64)  # SciPy prefers float64 internally
    rhs = RHS_REGISTRY[cfg.system](**(cfg.params or {}))

    def rhs_1d(t, y):
        # SciPy gives y shape (state_dim,). Our rhs expects (1, state_dim).
        dy = rhs(t, y[None, :])[0]
        return dy

    sol = solve_ivp(rhs_1d, (cfg.t0, cfg.t1), y0, t_eval=t_eval.astype(np.float64),
                    rtol=cfg.rtol, atol=cfg.atol, method=cfg.method)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    y_true = sol.y.T.astype(np.float32)   # (N, 2)
    y_noisy = y_true + (cfg.noise * rng.standard_normal(y_true.shape, dtype=np.float32)) if cfg.noise > 0 else y_true.copy()
    return t_eval, y_true, y_noisy, rhs
