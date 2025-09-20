import argparse
import numpy as np
from pathlib import Path
from node_25.viz.plotting import plot_predictions

def main():
    p = argparse.ArgumentParser("node25-viz-preds", description="Visualize predictions from an NPZ file.")
    p.add_argument("--npz", required=True, help="Path to predictions_seed0.npz (contains t, y_true, y_noisy, y_pred)")
    p.add_argument("--out", type=str, default=None, help="Where to save PNG (default next to NPZ)")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    data = np.load(args.npz)
    t = data["t"]; y_true = data["y_true"]; y_noisy = data.get("y_noisy"); y_pred = data["y_pred"]

    out = Path(args.out) if args.out else Path(args.npz).with_suffix(".png")
    title = f"Predictions vs ground truth\n{Path(args.npz).name}"
    plot_predictions(t, y_true, y_pred, y_noisy=y_noisy, title=title, save=out, show=args.show)

if __name__ == "__main__":
    main()
