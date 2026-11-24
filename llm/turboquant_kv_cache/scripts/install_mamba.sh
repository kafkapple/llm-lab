#!/usr/bin/env bash
set -euo pipefail

python -c 'import torch; assert torch.__version__.startswith("2.7."), torch.__version__'
wheel='https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
python -m pip install --no-deps "$wheel"
python -m pip install einops
python -c 'from mamba_ssm import Mamba; print("mamba-ssm import OK")'
