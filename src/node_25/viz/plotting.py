from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt


def _to_numpy(x: Any) -> np.ndarray:
    """Cnvert torch/jax/np arrays to numpy without importing heavy backends here."""
    if x is None:
        return None
    try:  # torch tensor
        import torch  # type: ignore
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:  # jax array
        import jax.numpy as jnp  # type: ignore
        if isinstance(x, jnp.ndarray):  # type: ignore[attr-defined]
            return np.asarray(x)
    except Exception:
        pass
    return np.asarray(x)


def plot_training_data(
    t,
    y_true: Optional[np.ndarray] = None,
    y_noisy: Optional[np.ndarray] = None,
    title: str = "Training data",
    labels: Tuple[str, str] = ("x", "v"),
    save: Optional[Path] = None,
    show: bool = False,
):
    """Overlay true & noisy trajectories."""
    t = _to_numpy(t)
    y_true = _to_numpy(y_true) if y_true is not None else None
    y_noisy = _to_numpy(y_noisy) if y_noisy is not None else None

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    if y_noisy is not None:
        ax.plot(t, y_noisy[:, 0], ".", ms=2, alpha=0.5, label=f"{labels[0]} noisy")
        ax.plot(t, y_noisy[:, 1], ".", ms=2, alpha=0.5, label=f"{labels[1]} noisy")
    if y_true is not None:
        ax.plot(t, y_true[:, 0], lw=1.8, label=f"{labels[0]} true")
        ax.plot(t, y_true[:, 1], lw=1.8, label=f"{labels[1]} true")
    ax.set_xlabel("t")
    ax.set_ylabel("state")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_predictions(
    t,
    y_true,
    y_pred,
    y_noisy: Optional[np.ndarray] = None,
    title: str = "Predictions vs Ground Truth",
    labels: Tuple[str, str] = ("x", "v"),
    save: Optional[Path] = None,
    show: bool = False,
    with_residuals: bool = True,
    with_phase: bool = True,
):
    """Compare y_pred to y_true; optionally show residuals and phase portrait."""
    t = _to_numpy(t)
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    y_noisy = _to_numpy(y_noisy) if y_noisy is not None else None

    nrows = 1 + int(with_residuals) + int(with_phase)
    height = 4.0 * nrows
    fig = plt.figure(figsize=(10, height))

    # Top: trajectories
    ax1 = fig.add_subplot(nrows, 1, 1)
    if y_noisy is not None:
        ax1.plot(t, y_noisy[:, 0], ".", ms=2, alpha=0.4, label=f"{labels[0]} noisy")
        ax1.plot(t, y_noisy[:, 1], ".", ms=2, alpha=0.4, label=f"{labels[1]} noisy")
    ax1.plot(t, y_true[:, 0], lw=1.8, label=f"{labels[0]} true")
    ax1.plot(t, y_true[:, 1], lw=1.8, label=f"{labels[1]} true")
    ax1.plot(t, y_pred[:, 0], lw=1.8, ls="--", label=f"{labels[0]} pred")
    ax1.plot(t, y_pred[:, 1], lw=1.8, ls="--", label=f"{labels[1]} pred")
    ax1.set_ylabel("state")
    ax1.set_title(title)
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend(ncols=2)

    r = y_pred - y_true
    row = 2

    if with_residuals:
        ax2 = fig.add_subplot(nrows, 1, row)
        ax2.plot(t, r[:, 0], lw=1.2, label=f"{labels[0]} residual")
        ax2.plot(t, r[:, 1], lw=1.2, label=f"{labels[1]} residual")
        ax2.axhline(0.0, color="k", lw=0.8, alpha=0.6)
        ax2.set_ylabel("residual")
        ax2.grid(True, ls="--", alpha=0.4)
        ax2.legend()
        row += 1

    if with_phase:
        ax3 = fig.add_subplot(nrows, 1, row)
        ax3.plot(y_true[:, 0], y_true[:, 1], lw=1.6, label="true")
        ax3.plot(y_pred[:, 0], y_pred[:, 1], lw=1.6, ls="--", label="pred")
        ax3.set_xlabel(labels[0])
        ax3.set_ylabel(labels[1])
        ax3.grid(True, ls="--", alpha=0.4)
        ax3.legend()

    fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
