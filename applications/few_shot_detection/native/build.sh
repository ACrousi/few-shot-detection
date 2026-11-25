#!/usr/bin/env bash
# Run inside the CUDA development container, without sudo.
set -euo pipefail
native_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${NATIVE_BUILD_DIR:-$native_dir/../build/native}"
target="${1:-gpu_geometry}"
case "$target" in
  all|gpu_geometry) ;;
  *) echo "Usage: bash $0 [all|gpu_geometry]" >&2; exit 2 ;;
esac
# Accept the old CUDA_ARCH=sm_110 spelling as well as CMake's numeric spelling.
architecture="${CUDA_ARCH:-110}"
cmake -S "$native_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${architecture#sm_}"
cmake --build "$build_dir" --parallel --target "$target"
