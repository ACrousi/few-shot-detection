#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
app="$root/applications/few_shot_detection"
source "$app/runtime_env.sh"
exec "$VIRTUAL_ENV/bin/python" -m few_shot_detection "$@"
