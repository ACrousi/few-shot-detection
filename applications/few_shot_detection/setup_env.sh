#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
platform="${1:-nvidia}"
case "$platform" in
  nvidia) exec bash "$project_dir/src/dinov3_cases/dinov3_cases/platforms/nvidia/setup_env.sh" ;;
  intel|qualcomm) echo "$platform: application model runtime is not implemented; no environment fallback" >&2; exit 2 ;;
  *) echo "Usage: $0 [nvidia|intel|qualcomm]" >&2; exit 2 ;;
esac
