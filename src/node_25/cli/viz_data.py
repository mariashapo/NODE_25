import argparse
from pathlib import Path
from node_25.core.datasets import SynthConfig, generate_synth
from node_25.viz.plotting import plot_training_data

def main():
    p = argparse.ArgumentParser("node25-viz-data", description="Visualize synthetic training data.")
    p.add_argument("--system", choices=["ho","vdp","do"], default="ho")
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float, default=10.0)
    p.add_argument("--out", type=str, default="results/viz/data.png")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    cfg = SynthConfig(system=args.system, N=args.N, t0=args.t0, t1=args.t1, y0=(0.0,1.0), noise=args.noise)
    t, y_true, y_noisy, _ = generate_synth(cfg)
    plot_training_data(t, y_true=y_true, y_noisy=y_noisy,
                       title=f"{args.system} - training data (noise={args.noise})",
                       save=Path(args.out), show=args.show)

if __name__ == "__main__":
    main()
