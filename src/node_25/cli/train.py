import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from torchdiffeq import odeint as odeint_torch

from node_25.core.datasets import SynthConfig, generate_synth
from node_25.trainers.pytorch_trainer import TrainConfig, train_one_seed
from node_25.eval.metrics import mean_ci_95

def to_torch(t_np, y_true_np, y_noisy_np, device="cpu"):
    t = torch.as_tensor(t_np, dtype=torch.float32, device=device)
    y_true = torch.as_tensor(y_true_np, dtype=torch.float32, device=device)
    y_noisy = torch.as_tensor(y_noisy_np, dtype=torch.float32, device=device)
    return t, y_true, y_noisy

def parse_widths(s: str):
    s = s.strip()
    return tuple(int(x) for x in s.split(",") if x) if s else ()

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def save_json(obj, path: Path): ensure_dir(path.parent); path.write_text(json.dumps(obj, indent=2))
def append_csv(row: dict, path: Path):
    ensure_dir(path.parent); write = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write: w.writeheader()
        w.writerow(row)

def choose_device(pref: str) -> str:
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()): return "cuda"
    if pref == "mps" or (pref == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()): return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser("node25-train", description="Synthetic Neural ODE runner (PyTorch)")
    p.add_argument("--system", choices=["ho","vdp","do"], default="vdp")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--widths", type=str, default="32,32")
    p.add_argument("--reg", type=float, default=0.001)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--method", type=str, default="dopri5")
    p.add_argument("--time-invariant", action="store_true")
    p.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="cpu")
    p.add_argument("--outdir", type=str, default="results/cs1")
    p.add_argument("--resample-data", action="store_true")
    p.add_argument("--save-preds", action="store_true")
    args = p.parse_args()

    device = choose_device(args.device)
    widths = parse_widths(args.widths)

    base_cfg = SynthConfig(system=args.system, N=200, t0=0.0, t1=15.0, y0=(0.0,1.0), noise=args.noise)
    t_np, y_true_np, y_noisy_np, _ = generate_synth(base_cfg)
    t, y_true, y_noisy = to_torch(t_np, y_true_np, y_noisy_np, device=device)
    y0 = y_true[0]

    mses = []
    first_seed_pred = None

    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        if args.resample_data:
            t_np, y_true_np, y_noisy_np, _ = generate_synth(base_cfg)
            t, y_true, y_noisy = to_torch(t_np, y_true_np, y_noisy_np, device=device)
            y0 = y_true[0]

        cfg = TrainConfig(lr=args.lr, epochs=args.epochs, widths=widths,
                          time_invariant=args.time_invariant, reg_lambda=args.reg,
                          rtol=args.rtol, atol=args.atol, method=args.method)
        out = train_one_seed(t, y_noisy, y0, cfg, device=device)
        mses.append(float(out["mse"]))
        print(f"Seed {seed} completed with MSE {float(out['mse'])}")

        if args.save_preds and first_seed_pred is None:
            with torch.no_grad():
                y_pred = odeint_torch(out["model"].rhs, y0, t, rtol=args.rtol, atol=args.atol, method=args.method)
            first_seed_pred = {
                "t": t.cpu().numpy(),
                "y_true": y_true.cpu().numpy(),
                "y_noisy": y_noisy.cpu().numpy(),
                "y_pred": y_pred.cpu().numpy(),
            }

    mean_mse, ci95 = mean_ci_95(mses)

    save_dir = Path(args.outdir) / args.system / f"pytorch_w{widths}_reg{args.reg}"
    save_json({
        "system": args.system, "device": device, "seeds": args.seeds, "epochs": args.epochs,
        "noise": args.noise, "widths": widths, "reg_lambda": args.reg,
        "time_invariant": bool(args.time_invariant), "resample_data": bool(args.resample_data),
        "rtol": args.rtol, "atol": args.atol, "method": args.method,
        "mse_mean": mean_mse, "mse_ci95": ci95, "mses": mses
    }, save_dir / "metrics.json")

    if first_seed_pred is not None:
        np.savez(save_dir / "predictions_seed0.npz", **first_seed_pred)

    append_csv({
        "system": args.system, "backend": "pytorch", "widths": str(widths), "reg": args.reg,
        "epochs": args.epochs, "noise": args.noise, "seeds": args.seeds,
        "time_invariant": int(args.time_invariant), "resample_data": int(args.resample_data),
        "device": device, "rtol": args.rtol, "atol": args.atol, "method": args.method,
        "mse_mean": f"{mean_mse:.8e}", "mse_ci95": f"{ci95:.8e}"
    }, Path(args.outdir) / "summary.csv")

    print(f"\n✅ {args.system} | widths={widths} | reg={args.reg} | seeds={args.seeds}")
    print(f"   MSE: {mean_mse:.4e} ± {ci95:.4e} (95% CI)")
    print(f"   Saved: {save_dir}/metrics.json and {Path(args.outdir)/'summary.csv'}")


if __name__ == "__main__":
    main()
