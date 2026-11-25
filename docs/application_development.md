# Application development

Generate an application with `rsctl init`. Each platform directory owns a complete
pipeline and optional ROS integration. Start from the CPU example to inspect packaging;
implement real platform runtimes directly in `platforms/nvidia`, `platforms/intel` or
`platforms/qualcomm`. There is no backend adapter interface to implement.

A platform executable accepts `--config PATH`, and `--check [--require-runtime]` for
validation. It owns all other flags, configuration fields, model loading, buffer lifetime,
thread safety and cleanup. Return a nonzero process exit code on unsupported runtimes.

Suite `pipeline-run` passes an absolute config path and forwards arguments after `--`.
Inside your platform module resolve model paths against the config directory. For an
in-process ROS application, directly import that application's concrete pipeline; this
is an application choice, not a Suite class contract.

Test complete observable results and error cleanup. Keep reference comparisons for
preprocessing, precision/quantization and device transfers. Real target validation is
required before claiming another platform supported.
