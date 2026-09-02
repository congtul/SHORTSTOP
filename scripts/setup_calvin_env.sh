#!/bin/bash
# One-environment setup for the CALVIN + MDT pipeline (Stage 7b), matching
# the sequence documented in docs/CALVIN_SETUP.md. Run from WSL2 Ubuntu,
# after `git submodule update --init --recursive` has pulled mdt_policy
# (and its own calvin_env/tacto submodules). Does NOT download any
# dataset/checkpoint -- see docs/CALVIN_SETUP.md sections 5-6 for that.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${1:-mdt_env}"

if [ ! -d "$REPO_ROOT/mdt_policy/calvin_env" ] || [ -z "$(ls -A "$REPO_ROOT/mdt_policy/calvin_env" 2>/dev/null)" ]; then
    echo "mdt_policy/calvin_env is empty -- run 'git submodule update --init --recursive' first." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^$ENV_NAME "; then
    conda create -y -n "$ENV_NAME" python=3.8
fi
conda activate "$ENV_NAME"

cd "$REPO_ROOT/mdt_policy/calvin_env/tacto"
pip install -e .
cd "$REPO_ROOT/mdt_policy/calvin_env"
pip install -e .

pip install setuptools==57.5.0
cd "$REPO_ROOT/mdt_policy/pyhash-0.9.3"
python setup.py build
python setup.py install

cd "$REPO_ROOT/mdt_policy"
pip install -r requirements.txt
pip install torchvision==0.15.2
pip install networkx==2.8.8

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda_libs.sh" <<EOF
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CONDA_PREFIX/lib/python3.8/site-packages/nvidia/cudnn/lib:\$LD_LIBRARY_PATH
EOF

bash "$REPO_ROOT/scripts/apply_mdt_patch.sh" || echo "Patch already applied or failed -- check manually if this is unexpected."

echo ""
echo "Done. Re-activate to pick up LD_LIBRARY_PATH: conda deactivate && conda activate $ENV_NAME"
echo "Next: docs/CALVIN_SETUP.md section 4 (smoke test) and section 5 (debug dataset)."
