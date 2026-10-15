#!/bin/bash
# One-time setup on macOS/Linux: creates .venv, installs dependencies,
# and runs a quick synthetic smoke test.  Usage:  bash setup.sh
set -e
cd "$(dirname "$0")"

# LightGBM on macOS needs OpenMP from Homebrew. Without it the harness
# falls back to scikit-learn's gradient boosting (slower, still works).
if [[ "$(uname)" == "Darwin" ]]; then
  if command -v brew >/dev/null 2>&1; then
    brew list libomp >/dev/null 2>&1 || brew install libomp
  else
    echo "Homebrew not found: skipping libomp. Install it from https://brew.sh for LightGBM."
  fi
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -c "import lightgbm; print('LightGBM', lightgbm.__version__)" \
  || echo "LightGBM unavailable: will fall back to scikit-learn."

python -m stuffplus.run_ablation --synthetic --variants A_tj_v3 --boot 50
echo
echo "Setup OK. Next: source .venv/bin/activate && python -m pipeline.fetch_data"
