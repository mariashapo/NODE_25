import argparse
from pathlib import Path
import torch

from node_25.pipelines.pytorch_experiment import (
    ExperimentConfig,
    PytorchExperiment,
    parse_widths,
)

def choose_device(pref: str) -> str:
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return "cuda"
    if pref == "mps" or (
        pref == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"

def parse_floats_list(s: str):
    s = s.strip()
    return tuple(float(x) for x in s.split(",") if x) if s else ()

def parse_ints_list(s: str):
    s = s.strip()
    return tuple(int(x) for x in s.split(",") if x) if s else ()

def main():
    p = argparse.ArgumentParser("node25-train", description="Synthetic Neural ODE runner (PyTorch)")
    # data/system
    p.add_argument("--system", choices=["ho", "vdp", "do"], default="vdp")
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float, default=15.0)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--resample-data", action="store_true")

    # model/train
    p.add_argument("--widths", type=str, default="32,32")
    p.add_argument("--reg", type=float, default=0.001)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--time-invariant", action="store_true")
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--method", type=str, default="dopri5")
    p.add_argument("--pretrain-fracs", type=str, default="")
    p.add_argument("--pretrain-epochs", type=str, default="")

    # execution
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--outdir", type=str, default="results/cs1/predictions")
    p.add_argument("--save-preds", action="store_true")

    # plots
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--plots-dir", type=str, default=None)

    args = p.parse_args()

    cfg = ExperimentConfig(
        system=args.system,
        N=args.N,
        t0=args.t0,
        t1=args.t1,
        y0=(0.0, 1.0),             # can be promoted to CLI later
        noise=args.noise,
        resample_data=bool(args.resample_data),

        widths=parse_widths(args.widths),
        reg=args.reg,
        lr=args.lr,
        epochs=args.epochs,
        time_invariant=bool(args.time_invariant),
        rtol=args.rtol,
        atol=args.atol,
        method=args.method,
        pretrain_fracs=parse_floats_list(args.pretrain_fracs),
        pretrain_epochs=parse_ints_list(args.pretrain_epochs),

        seeds=args.seeds,
        device=choose_device(args.device),
        outdir=Path(args.outdir),
        save_preds=bool(args.save_preds),

        save_plots=bool(args.save_plots),
        plots_dir=Path(args.plots_dir) if args.plots_dir else None,
    )

    exp = PytorchExperiment(cfg)
    exp.run()

if __name__ == "__main__":
    main()
