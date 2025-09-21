import argparse
from pathlib import Path
import numpy as np
from node_25.core.datasets import SynthConfig, generate_synth
from node_25.trainers.pyomo_trainer import PyomoTrainConfig, train_collocation

def parse_widths(s: str):
    s = s.strip()
    return tuple(int(x) for x in s.split(",") if x) if s else ()

def main():
    p = argparse.ArgumentParser("node25-train-pyomo", description="Pyomo collocation backend")
    p.add_argument("--system", choices=["ho","vdp","do"], default="ho")
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float, default=10.0)
    p.add_argument("--widths", type=str, default="32,32")
    p.add_argument("--act", type=str, default="tanh")
    p.add_argument("--reg", type=float, default=0.0)
    p.add_argument("--solver", type=str, default="ipopt")
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--outdir", type=str, default="results/cs1")
    args = p.parse_args()

    # Uniform-grid GT + noisy obs
    cfg = SynthConfig(system=args.system, N=args.N, t0=args.t0, t1=args.t1, y0=(0.0,1.0), noise=args.noise)
    t_u, y_true, y_noisy, _ = generate_synth(cfg)
    y0 = y_true[0].astype(np.float64)

    cfg_tr = PyomoTrainConfig(
        widths=parse_widths(args.widths),
        act=args.act,
        lambda_reg=args.reg,
        solver=args.solver,
        solver_options={"max_iter": args.max_iter},
    )
    out = train_collocation(t_u, y_noisy, y0, t0=args.t0, t1=args.t1, cfg=cfg_tr)

    # report + save
    mse_train = float(np.mean((y_true - out["y_pred_uniform"])**2))
    save_dir = Path(args.outdir) / args.system / f"pyomo_w{parse_widths(args.widths)}_reg{args.reg}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # y_obs at collocation nodes (2D interp)
    y_obs_nodes = np.column_stack([
        np.interp(out["t_nodes"], t_u, y_noisy[:, d]) for d in range(y_noisy.shape[1])
    ])

    np.savez(save_dir / "predictions_train_colloc.npz",
            t=out["t_nodes"], y_pred=out["y_pred_nodes"], y_obs=y_obs_nodes)

    np.savez(save_dir / "predictions_train_uniform.npz",
            t=t_u, y_true=y_true, y_noisy=y_noisy, y_pred=out["y_pred_uniform"])

    (save_dir / "metrics.txt").write_text(f"mse_train={mse_train:.8e}\n")
    print(f"✅ mse_train={mse_train:.4e}")
    print(f"Saved: {save_dir/'predictions_train_uniform.npz'}")


if __name__ == "__main__":
    main()
