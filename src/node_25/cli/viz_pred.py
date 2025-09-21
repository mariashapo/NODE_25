import argparse
import numpy as np
from pathlib import Path
from node_25.viz.plotting import plot_predictions

def main():
    p = argparse.ArgumentParser("node25-viz-preds", description="Visualize predictions from an NPZ file.")
    p.add_argument("--npz", required=True, help="Path to .npz (uniform or collocation).")
    p.add_argument("--out", type=str, default=None, help="Where to save PNG (default next to NPZ)")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    data = np.load(args.npz)
    t = data["t"]
    y_pred = data["y_pred"]

    # Auto-detect file type
    if "y_true" in data:
        # uniform-grid file: compare vs ground truth
        y_true = data["y_true"]
        y_noisy = data.get("y_noisy")
        title = f"Predictions vs Ground Truth (uniform)\n{Path(args.npz).name}"
    elif "y_obs" in data:
        # collocation file: compare vs observations at Chebyshev nodes
        y_true = data["y_obs"]   # use obs as the reference series
        y_noisy = None
        title = f"Predictions vs Observations (collocation nodes)\n{Path(args.npz).name}"
    else:
        raise KeyError("NPZ must contain either 'y_true' (uniform) or 'y_obs' (collocation).")

    out = Path(args.out) if args.out else Path(args.npz).with_suffix(".png")
    plot_predictions(t, y_true, y_pred, y_noisy=y_noisy, title=title, save=out, show=args.show)

if __name__ == "__main__":
    main()
