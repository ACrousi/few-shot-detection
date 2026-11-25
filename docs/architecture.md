# Platform-owned pipelines

Each platform implements and owns its complete pipeline: preprocessing, models,
postprocessing, memory, streams, transport, scheduling and cleanup. Suite does not
impose an InferenceBackend, ModelConfig, tensor schema, provider class or stage API.

```text
rsctl / application entry
          ↓ selects one executable module
platforms/<platform>/pipeline.py
          ↓ owns the complete computation
input → preprocessing → models → postprocessing → application results
```

`robotic_inference` contains only YAML loading and process launch helpers. The launch
configuration requires `platform` and `pipeline` (a trusted Python module name).
Every remaining field is interpreted and validated by the selected platform.
Relative artifact paths are resolved by that pipeline relative to its own config.
No model tensors pass through Suite and no subprocess is created per model stage.
Suite launches the platform executable once; that executable decides its own internals.

```yaml
platform: nvidia
pipeline: my_app.platforms.nvidia.pipeline
# Model artifacts, runtime settings and buffers below are platform-owned.
```

```bash
rsctl validate --config platform.yaml
rsctl validate --config platform.yaml --require-runtime
rsctl pipeline-run --config platform.yaml -- --image sample.png
```

`validate` invokes the selected executable with `--check`; `--require-runtime` requests
its actual model/runtime loading check. Checks and accuracy guarantees belong to that
implementation. `pipeline-run` returns the executable's exit code without fallback.
The platform program owns cleanup of its threads, subprocesses and devices.

`rsctl env`, `install`, `run`, `doctor` and build planning remain deployment utilities.
Environment checks report interpreter isolation, dependencies and architecture; they
cannot certify model execution. Platform recipes are environment hints, not pipelines.

The generated template contains a complete CPU mean-RGB **MOCK** example in
`platforms/cpu/pipeline.py`, with its own optional ROS node. NVIDIA/Intel/Qualcomm
entries are explicit scaffolds that fail until implemented; no fake successful inference.
Applications may share useful algorithms, but are not required to subclass common
interfaces or move device tensors through NumPy between stages.

## Breaking migration

Removed: InferenceBackend, TensorSpec, ModelConfig, Capabilities, backend registry,
ORT/OpenVINO/TensorRT/QNN/mock backend wrappers, PlatformProfile, PlatformProvider,
stage factories and tensor-schema overrides. Platform check now uses `rsctl validate`;
`platform-check` is removed. Old clients must implement their runtime inside a platform
pipeline and update imports/configuration. Existing validation reports are historical.
