#!/usr/bin/env bash
set -eo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --hardware ) ]]; then
  echo 'Usage: ./verify.sh [--hardware]' >&2; exit 2
fi
source /opt/ros/jazzy/setup.bash
app="$root/applications/few_shot_detection"
source "$app/runtime_env.sh"
python="$VIRTUAL_ENV/bin/python"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_DISABLE_PROGRESS_BARS=1
"$python" -m pytest -q -m 'not hardware'
"$python" -m pytest -q "$app/tests"
if [[ "${1:-}" == --hardware ]]; then
  export ROS_DOMAIN_ID=131 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
  "$python" "$app/tests/check_platform_runtime.py"
  "$python" "$app/tests/check_official_pipeline.py"
  "$python" "$app/tests/check_grounded_trt_pipeline.py"
  "$python" "$app/tests/check_async_kcf_pipeline.py"
fi
