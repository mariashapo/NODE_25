# node_25

Synthetic Neural ODE experiments with a clean CLI, mean ± 95% CI, and minimal dependencies.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'    # zsh: keep the quotes

node25-train --device cpu --system ho --seeds 1 --epochs 50 --widths 32,32 --noise 0.1
node25-train --device cpu --system vdp --seeds 1 --epochs 500 --widths 32 --noise 0.1

node25-viz-data --system ho --noise 0.1 --show
# or save:
node25-viz-data --system vdp --noise 0.05 --out results/viz/vdp_data.png

node25-viz-preds --npz results/cs1/ho/pytorch_w(32, 32)_reg0.0/predictions_seed0.npz --show
```