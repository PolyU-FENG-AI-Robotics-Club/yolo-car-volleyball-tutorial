#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_BIN" ]]; then
  echo 'Install Miniconda or Anaconda first, then reopen your terminal.' >&2
  exit 1
fi
if [[ ! -x .conda/bin/python ]]; then
  "$CONDA_BIN" env create --prefix "$PWD/.conda" --file environment.yml
fi
.conda/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.conda/bin/python -m pip install -r requirements.txt
.conda/bin/python prepare_data.py
.conda/bin/python check_environment.py
echo "Ready. Run: .conda/bin/python detect.py"
