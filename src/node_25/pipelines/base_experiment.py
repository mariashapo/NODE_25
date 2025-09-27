# src/node_25/pipelines/pytorch_experiment.py
from __future__ import annotations
from dataclasses import asdict, is_dataclass, dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Literal
from abc import ABC, abstractmethod
from node_25.pipelines.utils import *
from node_25.eval.metrics import mean_ci_95


Method = Literal["dopri5", "rk4", "euler"]  # extend as needed
Device = Literal["cpu", "cuda", "mps"]

@dataclass(frozen=True)
class ExperimentConfig:
    # data / system
    system: Literal["ho", "vdp", "do"] = "vdp"
    N: int = 200
    t0: float = 0.0
    t1: float = 15.0
    y0: Tuple[float, float] = (0.0, 1.0)
    noise: float = 0.1
    resample_data: bool = False

    # model / train (generic)
    widths: Tuple[int, ...] = (32, 32)
    reg: float = 1e-3
    lr: float = 1e-3
    epochs: int = 800
    time_invariant: bool = True
    rtol: float = 1e-3
    atol: float = 1e-4
    method: Method = "dopri5"
    pretrain_fracs: Tuple[float, ...] = ()
    pretrain_epochs: Tuple[int, ...] = ()

    # execution
    seeds: int = 1                 # or Tuple[int, ...] if you want multiple
    device: Device = "cpu"
    outdir: Path = Path("results/cs1")
    save_preds: bool = False

    # plots
    save_plots: bool = False
    plots_dir: Optional[Path] = None

    def validate(self) -> None:
        if self.pretrain_fracs and len(self.pretrain_fracs) != len(self.pretrain_epochs):
            raise ValueError("pretrain_fracs and pretrain_epochs must have the same length.")

# --------------------------- base sequential class ---------------------------

class ExperimentSequentialBase(ABC):
    """
    Reusable base for sequential (one-trajectory) Neural ODE experiments.
    Subclasses should:
      - set `backend` string
      - implement `_prepare_train_data(seed)` and `_train_seed(seed) -> Dict`
    The base provides:
      - filesystem-safe run_id / run_root (includes widths, reg, pretraining schedule)
      - aggregation + JSON/CSV saving
    """
    backend: str = "base"
    # Metric keys the base will aggregate with mean ± CI
    metric_keys: Tuple[str, ...] = (
        "mse_train_true", "mse_train_noisy", "mse_test_true", "mse_test_noisy"
    )

    def __init__(self, cfg):
        self.cfg = cfg
        self.run_id = self._build_run_id()
        self.run_root = cfg.outdir / cfg.system / self.run_id
        ensure_dir(self.run_root)
        self.on_init()

    def on_init(self) -> None:
        """Optional hook; subclasses can prepare initial data here."""
        pass

    def _build_run_id(self) -> str:
        wtag = widths_tag(getattr(self.cfg, "widths", ()))
        run_id = f"{self.backend}_w{wtag}"
        if hasattr(self.cfg, "reg"):
            run_id += f"_reg{getattr(self.cfg, 'reg'):g}"
        if hasattr(self.cfg, "epochs"):
            run_id += f"_ep{getattr(self.cfg, 'epochs'):g}"
        pttag = schedule_tag(
            getattr(self.cfg, "pretrain_fracs", ()),
            getattr(self.cfg, "pretrain_epochs", ()),
        )
        if pttag != "pt-none":
            run_id += f"_{pttag}"
        return run_id

    @abstractmethod
    def _prepare_train_data(self, seed: Optional[int]):
        """Prepare training trajectory (and store into self.* as needed)."""

    @abstractmethod
    def _train_seed(self, seed: int) -> Dict:
        """Train one seed, save per-seed artifacts, and return a metrics dict."""

    def _asdict_cfg(self):
        return asdict(self.cfg) if is_dataclass(self.cfg) else vars(self.cfg)

    def _summary_row(self, agg: Dict) -> Dict:
        """One-row CSV summary per run; subclasses can override if needed."""
        return {
            "system": getattr(self.cfg, "system", "unknown"),
            "backend": self.backend,
            "widths": str(getattr(self.cfg, "widths", ())),
            "reg": getattr(self.cfg, "reg", float("nan")),
            "epochs": getattr(self.cfg, "epochs", -1),
            "noise": getattr(self.cfg, "noise", float("nan")),
            "seeds": getattr(self.cfg, "seeds", -1),
            "time_invariant": int(getattr(self.cfg, "time_invariant", True)),
            "resample_data": int(getattr(self.cfg, "resample_data", False)),
            "device": getattr(self.cfg, "device", "cpu"),
            "rtol": getattr(self.cfg, "rtol", float("nan")),
            "atol": getattr(self.cfg, "atol", float("nan")),
            "method": getattr(self.cfg, "method", ""),
            "mse_train_true_mean": f"{agg.get('mse_train_true_mean', float('nan')):.8e}",
            "mse_train_true_ci95": f"{agg.get('mse_train_true_ci95', float('nan')):.8e}",
            "mse_test_true_mean": f"{agg.get('mse_test_true_mean', float('nan')):.8e}",
            "mse_test_true_ci95": f"{agg.get('mse_test_true_ci95', float('nan')):.8e}",
        }

    def run(self) -> Dict:
        per_seed: List[Dict] = []
        for seed in range(getattr(self.cfg, "seeds", 1)):
            m = self._train_seed(seed)
            per_seed.append(m)
            # Compact per-seed print
            msg = (
                f"Seed {seed}: "
                f"train_true={m.get('mse_train_true', float('nan')):.3e}, "
                f"train_noisy={m.get('mse_train_noisy', float('nan')):.3e} | "
                f"test_true={m.get('mse_test_true', float('nan')):.3e}, "
                f"test_noisy={m.get('mse_test_noisy', float('nan')):.3e}"
            )
            print(msg)

        # Aggregate
        agg: Dict = {
            "system": getattr(self.cfg, "system", "unknown"),
            "backend": self.backend,
            "device": getattr(self.cfg, "device", "cpu"),
            "seeds": getattr(self.cfg, "seeds", 1),
            "epochs": getattr(self.cfg, "epochs", -1),
            "noise": getattr(self.cfg, "noise", 0.0),
            "widths": getattr(self.cfg, "widths", ()),
            "reg_lambda": getattr(self.cfg, "reg", 0.0),
            "time_invariant": bool(getattr(self.cfg, "time_invariant", True)),
            "rtol": getattr(self.cfg, "rtol", 1e-3),
            "atol": getattr(self.cfg, "atol", 1e-4),
            "method": getattr(self.cfg, "method", ""),
            "run_id": self.run_id,
            "config": self._asdict_cfg(),
            "per_seed": per_seed,
        }
        for k in self.metric_keys:
            series = [m[k] for m in per_seed if k in m]
            mean_, ci_ = mean_ci_95(series) if series else (float("nan"), float("nan"))
            agg[k] = series
            agg[f"{k}_mean"] = mean_
            agg[f"{k}_ci95"] = ci_

        save_json(agg, self.run_root / "metrics.json")

        # One-row summary per run
        summary_row = self._summary_row(agg)
        append_csv(summary_row, getattr(self.cfg, "outdir", Path("results")) / "summary.csv")

        print("\n✅ Run complete")
        if "mse_train_true_mean" in agg and "mse_test_true_mean" in agg:
            print(
                f"   train_true = {agg['mse_train_true_mean']:.4e} ± {agg['mse_train_true_ci95']:.4e} | "
                f"test_true  = {agg['mse_test_true_mean']:.4e} ± {agg['mse_test_true_ci95']:.4e}"
            )
        print(f"   Saved run metrics: {self.run_root/'metrics.json'}")
        return agg


