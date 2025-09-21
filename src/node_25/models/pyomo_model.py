# src/node_25/models/pyomo_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyomo.environ as pyo


@dataclass
class CollocConfig:
    """Configuration for the collocation-based Neural ODE (Pyomo)."""
    widths: Sequence[int] = (32, 32)   # hidden layer widths, output is inferred as state dim
    act: str = "tanh"                  # 'tanh' | 'sigmoid' | 'relu'
    lambda_reg: float = 0.0            # L2 regularization on weights & biases
    fix_y0: bool = True                # enforce Y[0] = y0
    time_invariant: bool = True        # if False, NN input is [t, y]; else input is y


_ACT = {
    "tanh": pyo.tanh,
    "sigmoid": lambda z: 1.0 / (1.0 + pyo.exp(-z)),
    # ReLU is non-smooth; IPOPT prefers smooth activations but we keep it available.
    "relu": lambda z: pyo.max_expression(0.0, z),
}


class NeuralODEPyomo:
    """
    Direct collocation Neural ODE:
      - Decision vars: Y[i,s] (states at collocation nodes t[i]) + NN parameters
      - Constraints:   (D @ Y)[i, s] = fθ(input_i)[s]  for all nodes i, states s
      - Objective:     (1/N) * Σ ||Y - Y_obs||^2  +  λ * (||W||^2 + ||b||^2)
    """

    def __init__(
        self,
        t: np.ndarray,              # shape (N,)
        y_obs: np.ndarray,          # shape (N, S)
        D: np.ndarray,              # shape (N, N) differentiation matrix for nodes t
        y0: np.ndarray,             # shape (S,)
        cfg: CollocConfig,
    ) -> None:
        t = np.asarray(t, dtype=np.float64)
        y_obs = np.asarray(y_obs, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)
        y0 = np.asarray(y0, dtype=np.float64)

        assert t.ndim == 1, "t must be 1D"
        assert y_obs.ndim == 2 and y_obs.shape[0] == t.size, "y_obs shape must be (N, S)"
        assert D.shape == (t.size, t.size), "D must be (N, N)"
        assert y0.ndim == 1, "y0 must be 1D"

        self.t = t
        self.y_obs = y_obs
        self.D = D
        self.y0 = y0
        self.cfg = cfg

        self.N = t.size
        self.S = y0.size

        # network sizes: input -> hidden widths -> output
        in_dim = self.S if cfg.time_invariant else (self.S + 1)  # add time if time-variant
        self._layer_sizes: List[int] = [in_dim, *cfg.widths, self.S]

        # pyomo model holder
        self.model: Optional[pyo.ConcreteModel] = None

    # ---------- build ----------

    def build(self) -> None:
        m = pyo.ConcreteModel(name="NeuralODE_Collocation")
        N, S = self.N, self.S
        Lsizes = self._layer_sizes
        L = len(Lsizes) - 1  # number of Dense layers

        # Sets
        m.I = pyo.RangeSet(0, N - 1)
        m.S = pyo.RangeSet(0, S - 1)
        m.L = pyo.RangeSet(0, L - 1)

        # Data as Params
        m.D = pyo.Param(m.I, m.I, initialize=lambda _m, i, j: float(self.D[i, j]), mutable=False)
        m.Yobs = pyo.Param(m.I, m.S, initialize=lambda _m, i, s: float(self.y_obs[i, s]), mutable=False)
        m.Y0 = pyo.Param(m.S, initialize=lambda _m, s: float(self.y0[s]), mutable=False)
        if not self.cfg.time_invariant:
            m.ti = pyo.Param(m.I, initialize=lambda _m, i: float(self.t[i]), mutable=False)

        # --- STATES: good initial guess + bounds from data ---
        y_min = float(self.y_obs.min() - 2.0 * self.y_obs.std())
        y_max = float(self.y_obs.max() + 2.0 * self.y_obs.std())
        m.Y = pyo.Var(
            m.I, m.S,
            initialize=lambda _m, i, s: float(self.y_obs[i, s]),
            bounds=lambda _m, i, s: (y_min, y_max)
        )

        # RHS at nodes
        m.F = pyo.Var(m.I, m.S, initialize=0.0)

        # --- NN params: small random init (not zeros) ---
        rng = np.random.default_rng(42)
        m.W_layers: Dict[int, pyo.Var] = {}
        m.b_layers: Dict[int, pyo.Var] = {}
        for l in range(L):
            out_dim = Lsizes[l + 1]
            in_dim = Lsizes[l]
            W0 = rng.standard_normal((out_dim, in_dim)) * 0.1
            b0 = rng.standard_normal(out_dim) * 0.1
            W = pyo.Var(range(out_dim), range(in_dim),
                        initialize=lambda _m, u, k, W0=W0: float(W0[u, k]),
                        bounds=(-100.0, 100.0))
            b = pyo.Var(range(out_dim),
                        initialize=lambda _m, u, b0=b0: float(b0[u]),
                        bounds=(-100.0, 100.0))
            setattr(m, f"W{l}", W)
            setattr(m, f"b{l}", b)
            m.W_layers[l] = W
            m.b_layers[l] = b

        # Hidden activations
        m.A_layers: Dict[int, pyo.Var] = {}
        for l in range(L - 1):
            out_dim = Lsizes[l + 1]
            A = pyo.Var(range(N), range(out_dim), initialize=0.0)
            setattr(m, f"A{l}", A)
            m.A_layers[l] = A

        # Activation
        act = _ACT.get(self.cfg.act, pyo.tanh)

        # Affine helper
        def _affine(W: pyo.Var, b: pyo.Var, vec, out_dim: int, in_dim: int):
            return [sum(W[u, k] * vec[k] for k in range(in_dim)) + b[u] for u in range(out_dim)]

        # Forward pass constraints
        m.forward = pyo.ConstraintList()
        for i in range(N):
            if self.cfg.time_invariant:
                vec = [m.Y[i, s] for s in range(S)]
            else:
                vec = [m.ti[i]] + [m.Y[i, s] for s in range(S)]

            for l in range(L - 1):
                out_dim = Lsizes[l + 1]; in_dim = Lsizes[l]
                z = _affine(m.W_layers[l], m.b_layers[l], vec, out_dim, in_dim)
                for u in range(out_dim):
                    m.forward.add(m.A_layers[l][i, u] == act(z[u]))
                vec = [m.A_layers[l][i, u] for u in range(out_dim)]

            out_dim = Lsizes[L]; in_dim = Lsizes[L - 1]
            z_out = _affine(m.W_layers[L - 1], m.b_layers[L - 1], vec, out_dim, in_dim)
            for s in range(S):
                m.forward.add(m.F[i, s] == z_out[s])

        # Collocation constraints: D @ Y = F
        def _colloc_rule(_m, i, s):
            return sum(_m.D[i, j] * _m.Y[j, s] for j in _m.I) == _m.F[i, s]
        m.colloc = pyo.Constraint(m.I, m.S, rule=_colloc_rule)

        # Initial condition (optional)
        if self.cfg.fix_y0 and N > 0:
            m.ic = pyo.Constraint(m.S, rule=lambda _m, s: _m.Y[0, s] == _m.Y0[s])

        # Objective: MSE + L2 reg
        def _obj_fit(_m):
            return (1.0 / N) * sum((_m.Y[i, s] - _m.Yobs[i, s]) ** 2 for i in _m.I for s in _m.S)

        def _obj_reg(_m):
            lam = float(self.cfg.lambda_reg)
            if lam <= 0.0:
                return 0.0
            reg = 0.0
            for l in range(L):
                out_dim = Lsizes[l + 1]; in_dim = Lsizes[l]
                W = getattr(_m, f"W{l}"); b = getattr(_m, f"b{l}")
                reg += sum(W[u, k] ** 2 for u in range(out_dim) for k in range(in_dim))
                reg += sum(b[u] ** 2 for u in range(out_dim))
            return lam * reg

        m.obj = pyo.Objective(expr=_obj_fit(m) + _obj_reg(m), sense=pyo.minimize)
        self.model = m

    # ---------- solve ----------

    def solve(self, tee: bool = True, solver: str = "ipopt", options: Optional[Dict[str, float]] = None):
        """Solve the NLP with the given solver and options (e.g., {'max_iter': 3000})."""
        assert self.model is not None, "Call build() before solve()."
        opt = pyo.SolverFactory(solver)
        if options:
            for k, v in options.items():
                opt.options[k] = v
        return opt.solve(self.model, tee=tee)

    # ---------- extract ----------

    def extract(self) -> Dict[str, np.ndarray | List[np.ndarray]]:
        """Return collocation times, predicted states, NN weights and biases."""
        m = self.model
        assert m is not None, "No model to extract from. Call build() + solve() first."

        N, S = self.N, self.S
        Y = np.array([[pyo.value(m.Y[i, s]) for s in range(S)] for i in range(N)], dtype=np.float64)
        F = np.array([[pyo.value(m.F[i, s]) for s in range(S)] for i in range(N)], dtype=np.float64)

        weights: List[np.ndarray] = []
        biases: List[np.ndarray] = []
        Lsizes = self._layer_sizes
        L = len(Lsizes) - 1
        for l in range(L):
            out_dim = Lsizes[l + 1]
            in_dim = Lsizes[l]
            Wl = np.array([[pyo.value(getattr(m, f"W{l}")[u, k]) for k in range(in_dim)]
                           for u in range(out_dim)], dtype=np.float64)
            bl = np.array([pyo.value(getattr(m, f"b{l}")[u]) for u in range(out_dim)], dtype=np.float64)
            weights.append(Wl)
            biases.append(bl)

        return {
            "t": self.t.copy(),
            "Y": Y,
            "F": F,
            "weights": weights,
            "biases": biases,
        }

    # ---------- numpy forward + scipy rollout for test ----------

    def _weights_np(self):
        """Extract weights/biases as NumPy arrays."""
        m = self.model; assert m is not None
        Lsizes = self._layer_sizes
        L = len(Lsizes) - 1
        W, b = [], []
        for l in range(L):
            out_dim, in_dim = Lsizes[l + 1], Lsizes[l]
            Wl = np.array([[pyo.value(getattr(m, f"W{l}")[u, k]) for k in range(in_dim)]
                           for u in range(out_dim)], dtype=np.float64)
            bl = np.array([pyo.value(getattr(m, f"b{l}")[u]) for u in range(out_dim)], dtype=np.float64)
            W.append(Wl); b.append(bl)
        return W, b

    def _act_np(self, z: np.ndarray) -> np.ndarray:
        if self.cfg.act == "tanh":
            return np.tanh(z)
        elif self.cfg.act == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        elif self.cfg.act == "relu":
            return np.maximum(0.0, z)
        else:
            # default safe
            return np.tanh(z)

    def _rhs_numpy(self, t: float, y: np.ndarray, Wb=None) -> np.ndarray:
        """dy/dt = f_theta(t,y) with NumPy (used by SciPy solve_ivp)."""
        if Wb is None:
            W, b = self._weights_np()
        else:
            W, b = Wb
        # build input vector
        x = np.asarray(y, dtype=np.float64)
        if not self.cfg.time_invariant:
            x = np.concatenate(([float(t)], x), axis=0)
        # forward through MLP
        L = len(W)
        h = x
        for l in range(L - 1):
            h = self._act_np(W[l] @ h + b[l])
        out = W[L - 1] @ h + b[L - 1]
        return out

    def simulate_scipy(self, y0: np.ndarray, t_eval: np.ndarray,
                       rtol: float = 1e-7, atol: float = 1e-9, method: str = "RK45") -> np.ndarray:
        """Roll out the learned ODE on t_eval using SciPy."""
        from scipy.integrate import solve_ivp
        Wb = self._weights_np()
        y0 = np.asarray(y0, dtype=np.float64)
        t_eval = np.asarray(t_eval, dtype=np.float64)
        sol = solve_ivp(lambda tt, yy: self._rhs_numpy(tt, yy, Wb),
                        (float(t_eval[0]), float(t_eval[-1])),
                        y0,
                        t_eval=t_eval,
                        rtol=rtol, atol=atol, method=method)
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        return sol.y.T  # (N, S)
