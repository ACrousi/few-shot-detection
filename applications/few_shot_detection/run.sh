#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Use the same large-image transport in camera and inference processes.
# Keep explicit caller-provided DDS profiles.
if [[ -z "${FASTDDS_DEFAULT_PROFILES_FILE:-}" && -z "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]]; then
  export FASTRTPS_DEFAULT_PROFILES_FILE="$script_dir/config/fastdds_rgbd.xml"
fi

set +u
source /opt/ros/jazzy/setup.bash
set -u
source "$script_dir/runtime_env.sh"
if [[ ! -x "$VIRTUAL_ENV/bin/python" ]]; then
  printf '%s\n' 'Python environment missing. Run setup_env.sh first.' >&2
  exit 1
fi
# This pipeline communicates only with ROS nodes on this machine.
# Override inherited SUBNET settings and remove explicit remote-discovery paths.
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_STATIC_PEERS ROS_DISCOVERY_SERVER ROS_LOCALHOST_ONLY
exec "$VIRTUAL_ENV/bin/python" "$script_dir/launch_pipeline.py" "$@"
