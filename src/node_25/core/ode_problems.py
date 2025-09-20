from typing import Callable, Dict
import numpy as np

Array = np.ndarray

def ho_rhs(omega2: float = 2.0) -> Callable[[float, Array], Array]:
    """Harmonic oscillator: x' = v, v' = -omega^2 * x"""
    def f(t: float, y: Array) -> Array:
        x, v = y[..., 0], y[..., 1]
        return np.stack([v, -omega2 * x], axis=-1)
    return f

def vdp_rhs(mu: float = 1.0, omega: float = 1.0) -> Callable[[float, Array], Array]:
    """Van der Pol: x' = v, v' = mu*(1 - x^2)*v - omega^2*x"""
    def f(t: float, y: Array) -> Array:
        x, v = y[..., 0], y[..., 1]
        return np.stack([v, mu*(1 - x**2)*v - (omega**2)*x], axis=-1)
    return f

def damped_rhs(damping: float = 0.1, omega2: float = 1.0) -> Callable[[float, Array], Array]:
    """Damped oscillator: x' = v, v' = -damping*v - omega^2*x"""
    def f(t: float, y: Array) -> Array:
        x, v = y[..., 0], y[..., 1]
        return np.stack([v, -damping*v - omega2*x], axis=-1)
    return f

RHS_REGISTRY: Dict[str, Callable[..., Callable[[float, Array], Array]]] = {
    "ho": ho_rhs, "vdp": vdp_rhs, "do": damped_rhs,
}
