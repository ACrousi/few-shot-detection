> Historical results from the previous backend architecture. Current execution is described in [architecture.md](architecture.md).

# Historical validation record — 2026-11-13

This section records the pre-refactor checkout. The Medicine example and its installation
checks were removed on 2026-11-16; historical results below are not current coverage.

Environment: Linux aarch64, Python 3.12.3. Development venv: NumPy 2.5.4, PyYAML 6.0.3,
pytest 9.2.0, OpenVINO 2026.4.1. ROS Jazzy smoke uses the system Python/NumPy 1.26.4 ABI.

- SDK and CLI installed independently as editable distributions, then built as wheels.
- Default suite: 40 passed, 2 explicitly skipped proprietary execution tests.
- OpenVINO creates a tiny local IR with sum/difference outputs, validates two named inputs,
  dynamic batches of 2 and 5, dtype/name failures, output ownership and cleanup.
- A generated `sample_vision` application imports and runs without ROS and its own test passes.
- Medicine mock runs with three independent instances; recognition and pose remain absent.
- `verified_perception` generated under `/tmp/robotic-suite-ros-validation/src`, built with
  colcon against ROS Jazzy (one package succeeded), and passed a live Image -> String round trip.
- Live ROS test uses domain 173 / localhost discovery; message frame ID and numerical mean
  were asserted. No camera or remote ROS graph was required.
- Host pytest plugin autoload was disabled because unrelated preinstalled ROS plugins were
  not complete inside the venv. A multiprocessing/fork deprecation warning remains in
  runtime discovery; no test failures occurred.

Repeat the ROS check after generating/building your package and sourcing its install:
```bash
# From the suite root, with system ROS Python and generated package on PYTHONPATH:
export PYTHONPATH="$PWD/packages/robotic_inference_sdk/src:${PYTHONPATH:-}"
ROS_DOMAIN_ID=173 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST python3 tests/ros_smoke.py --package my_perception_demo --config ros_ws/src/my_perception_demo/config/default.yaml
```
Hardware tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -m hardware -v`.
OpenVINO skips if absent; TensorRT/QNN always skip with an explicit incomplete-implementation
reason, while separate default tests assert their failure behavior. These skips are not
hardware certification. No real TensorRT engine, QNN graph, production model, GPU/NPU,
Docker image build or Humble installation was tested.

## Scheme A revision
- Regression suite: 49 passed, 2 explicit hardware skips.
- Actual `env create --mock` produced isolated venvs with no system site packages; SDK/pip
  checks and dependency snapshots succeeded even from a sourced ROS shell.
- `install --source` / `run` executed the Medicine example in its selected venv.
- `install --ros-workspace` built a generated package with system Jazzy colcon, then `run`
  executed its `__main__` using a different isolated venv, without ROS inference imports.
- Tests verify inherited PYTHONPATH removal, argument/exit-code forwarding, existing-path
  protection, moved/incomplete environment rejection and one application per environment.
- Platform requirement recipes are not certified target locks. TensorRT/QNN remain incomplete.


## Multi-demo / ORT revision — 2026-11-16

- Removed the Medicine application source/configuration and its application-specific test.
  CI installs only SDK and CLI; CLI configuration tests use the generic template.
- Added ORT adapter and optional `ort` dependency; real CPU tests generate a local ONNX
  Add/Sub graph (no downloaded model), exercise named multi-I/O, changing batch size,
  dtype/shape/name rejection, output ownership, close/reload and failed-load cleanup.
- Tested missing ORT, absent requested EP, invalid options and configured contract mismatch.
- Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`.
  Result: **57 passed, 2 skipped**, one pre-existing OpenVINO discovery/fork warning.
- Runtime: Python 3.12, ONNX Runtime 1.32.0, ONNX 1.23.2, Linux aarch64.
- `pip check`: no broken requirements. Generated-package pipeline tests passed as part
  of the regression suite. ROS live round trip and Docker builds were not rerun.
- TensorRT/QNN native execution remains incomplete and explicitly skipped. ORT accelerator
  EPs, device placement, strict no-CPU execution on accelerator hardware, and production
  model accuracy/performance were not tested. CPU execution is not target certification.
- Pre-edit source backup (untracked checkout):
  `/tmp/robotic-suite-before-refactor-1794808901.tar.gz` (temporary local recovery artifact).
