# node_25

Synthetic Neural ODE experiments with a clean CLI, mean ± 95% CI, and minimal dependencies.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . # installs only the core dependencies
python -m pip install -e '.[dev]'    # zsh: keep the quotes
python -m pip install -e '.[pyomo]'

## quick solver check
python - <<'PY'
import pyomo.environ as pyo
for s in ["ipopt","cbc","glpk"]:
    print(s, "available:", pyo.SolverFactory(s).available())
PY

# --------------------------------------------------------------- PYTORCH ---------------------------------------------------------------
node25-train-pytorch --device cpu --system ho --seeds 1 --epochs 50 --widths 32,32 --noise 0.1
node25-train-pytorch --device cpu --system vdp --seeds 1 --epochs 500 --widths 32 --noise 0.1

# show data
node25-viz-data --system ho --noise 0.1 --show
# or save:
node25-viz-data --system vdp --noise 0.05 --out results/viz/vdp_data.png


# pytorch with pre-training
node25-train-pytorch --device cpu --system ho --seeds 1 --epochs 500 \
  --widths 32 --noise 0.1 \
  --pretrain-fracs "0.1" --pretrain-epochs "500"


node25-train-pytorch --device cpu --system vdp --seeds 1 --epochs 1000 \
  --widths 32 --noise 0.1 --save-preds \
  --pretrain-fracs "0.2" --pretrain-epochs "500"

# save results
node25-viz-preds --npz results/cs1/ho/pytorch_w(32, 32)_reg0.0/predictions_seed0.npz --show
node25-viz-preds --npz 'results/cs1/vdp/pytorch_w(32,)_reg0.001/predictions_seed0.npz' --out results/viz/vdp_pytorch_training.png

# ----------------------------------------------------------------- PYOMO ---------------------------------------------------------------  

node25-train-pyomo --system ho --N 150 --t0 0 --t1 10 --widths 32 \
  --noise 0.1 --reg 1e-6 --solver ipopt --max-iter 3000
python -m pip install -e .
node25-train-pyomo --system vdp --N 150 --t0 0 --t1 10 --widths 32 \
  --noise 0.1 --reg 1e-6 --solver ipopt --max-iter 3000

# saving results
node25-viz-preds --npz 'results/cs1/ho/pyomo_w(32,)_reg1e-06/predictions_train_uniform.npz' --out results/viz/ho_pyomo_training.png 
node25-viz-preds --npz 'results/cs1/vdp/pyomo_w(32,)_reg1e-06/predictions_train_uniform.npz' --out results/viz/vdp_pyomo_training.png 



```