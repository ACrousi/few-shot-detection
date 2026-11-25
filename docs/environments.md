# Scheme A: application deployment environments

Scheme A is the default. ROS packages/colcon handle packaging and installation; rosdep
handles known system dependencies; a normal Python venv holds one application's AI Python
dependencies. The application package never creates its own environment. Multiple model
instances within an application share this environment; conflicting dependency versions
require separate processes/environments. No IPC framework is introduced.

## Install deployment tools once
From the suite root:
```bash
python3 -m venv .tools-venv
.tools-venv/bin/python -m pip install ./packages/robotic_inference_sdk ./tools/rsctl
export PATH="$PWD/.tools-venv/bin:$PATH"
```
The existing `.venv` in the development checkout can also run rsctl. The tooling environment
is separate from deployment environments. `env create` never overwrites an existing directory.

## Standalone Python route: no ROS required
```bash
rsctl init --name my_perception_demo --destination generated
rsctl env create --platform intel --path .envs/perception --sdk packages/robotic_inference_sdk --mock
rsctl install my_perception_demo --venv .envs/perception --source generated/my_perception_demo
rsctl env check --path .envs/perception
rsctl run my_perception_demo --venv .envs/perception -- --config "$PWD/generated/my_perception_demo/config/default.yaml"
```
This creates a real isolated venv, installs the SDK and selected requirements, records a
resolved dependency snapshot, installs the Python application, and invokes that venv's Python.
`--mock` is explicit development mode: it selects a CPU demonstration recipe, not a hidden fallback
for failed vendor runtimes. It does not edit application pipeline configuration.

For an actual Intel environment omit `--mock`:
```bash
rsctl env create --platform intel --path .envs/intel-app --python /usr/bin/python3 --sdk packages/robotic_inference_sdk
```
Profiles select requirements recipes in `deployment/requirements/`. Intel installs OpenVINO;
NVIDIA and Qualcomm do not automatically pip-install proprietary runtime packages. Provision
matching JetPack/TensorRT/CUDA or licensed QNN/QAIRT dependencies separately, then rerun
`env check` for dependencies/isolation, then run the application platform check using
`rsctl validate --config platform.yaml --require-runtime`. Environment creation does not
certify accelerator support; runtime and model validation belong to the application.

Use `--requirements /absolute/path/to/target.lock.txt` to replace the recipe with a target-
validated requirements file. Relative includes resolve against that original file. `--sdk`
accepts a local SDK source directory or wheel, avoiding reliance on a nonexistent public
release. `--python` chooses the interpreter; it must meet the profile's Python minimum.

## ROS packaging route: separate build and inference steps
Use the system interpreter compatible with your ROS installation. From the suite root:
```bash
source /opt/ros/jazzy/setup.bash
rsctl init --name my_perception_demo --destination robot_ws/src
cd robot_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select my_perception_demo
cd ..
rsctl env create --platform intel --path .envs/perception --python /usr/bin/python3 --sdk packages/robotic_inference_sdk --mock
# This command independently invokes system colcon and registers its installed package.
rsctl install my_perception_demo --venv .envs/perception --ros-workspace robot_ws
rsctl run my_perception_demo --venv .envs/perception -- --config "$PWD/robot_ws/src/my_perception_demo/config/default.yaml"
```
The initial manual colcon build illustrates independence and may be omitted if `rsctl install
--ros-workspace` will perform the build. rosdep remains an explicit step: rsctl does not run
sudo or invent keys for pip dependencies. The default isolated colcon install layout is
supported; merged installs are not yet supported by the registration command. It verifies
there is one matching source package, a matching Python install directory, and `__main__.py`.
Application AI dependencies belong in the venv recipe; colcon does not pip-install them.

Direct execution is also possible after sourcing the workspace:
```bash
source robot_ws/install/setup.bash
.envs/perception/bin/python -m my_perception_demo --config robot_ws/src/my_perception_demo/config/default.yaml
```
Direct invocation inherits PYTHONPATH from the sourced shell. For stronger isolation use
`rsctl run`, which ignores inherited PYTHONPATH/user-site and adds only the registered
application's colcon install directory. It invokes `venv/bin/python -I` explicitly; activation
and generated `ros2 run` shebangs do not control its interpreter. CLI arguments go after `--`;
the process exit code is propagated. Native runtime variables such as library paths remain
available because venv is not OS/driver isolation.

Optional ROS launch uses the explicit interpreter and isolated Python as well:
```bash
source robot_ws/install/setup.bash
ros2 launch my_perception_demo perception.launch.py venv_python:="$PWD/.envs/perception/bin/python"
```
This launches the standalone demonstration, prints one mock summary and exits. It is not a
ROS camera subscriber. The old in-process ROS node remains available via
`perception_node.launch.py`, with its original ROS/runtime ABI requirements (Scheme B).
No interprocess camera bridge or broker is implied by Scheme A.

## State, checks and rebuilds
Each environment contains `robotic-suite-env.json`, `COLCON_IGNORE`, the requested recipe
snapshot and `requirements.lock.txt` from `pip freeze --all`. Creation failures leave a
marked incomplete environment for inspection; create a new path after correcting dependencies.
One app may be registered per environment. Existing environments are never deleted automatically.
`env check` runs inside the selected venv, checking isolation, pip dependencies, target
architecture. Runtime execution is reported as not verifiable by environment inventory. It does not require ROS and does not claim model
inference validation. `env create/check` exit 1 when checks fail, 2 for command/configuration
errors. An environment may be successfully created while its target runtime check fails.

Retain the recipe, interpreter/platform/SDK metadata and resolved snapshot outside the venv
before delivery. For example:
```bash
mkdir -p deployment-locks/perception
cp .envs/perception/requirements.lock.txt .envs/perception/requirements.requested.txt .envs/perception/robotic-suite-env.json deployment-locks/perception/
```
Snapshots may contain local source URLs; replace those with retained wheel locations before
replay. They are not hashed cross-platform lockfiles. Recipes use version ranges, not a claim
of universal compatibility. Resolve, validate and archive exact wheels/locks per target.
Venvs cannot be moved across machines; rsctl rejects relocated managed environments. Rebuild
on the target with preserved dependencies. CUDA, drivers, system libraries and Qualcomm SDKs
are managed separately, not isolated by venv.

References: [Python venv](https://docs.python.org/3.12/library/venv.html),
[ROS Python package guidance](https://docs.ros.org/en/jazzy/How-To-Guides/Using-Python-Packages.html).

## Platform runtime checks

Platform recipes only select installation requirements. They do not select or probe a
model backend. Each application platform module owns runtime loading and execution;
run its `--check --require-runtime` command in the application's interpreter, or use
`rsctl validate --config platform.yaml --require-runtime` from that environment.
