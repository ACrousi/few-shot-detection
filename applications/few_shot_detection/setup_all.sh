#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo '[1/3] Setting up the model/application Python environment...'
bash "$project_dir/setup_env.sh" "${1:-nvidia}"
echo '[2/3] Checking CPU KCF support...'
"${DINOV3_VENV:-$project_dir/.venv}/bin/python" -c 'from dinov3_cases.tracking.kcf import check_kcf; check_kcf()'
echo '[3/3] Building CUDA geometry for the current NVIDIA model deployment...'
bash "$project_dir/native/build.sh" gpu_geometry
