"""Launch one complete platform executable. No tensor or stage APIs.

The module is trusted application code. Suite passes the absolute configuration
path; the module owns its config, runtime, data movement and pipeline lifecycle.
"""
from pathlib import Path
import subprocess
import sys
from .configuration import load_pipeline_config
from .errors import ConfigurationError


def run_pipeline(config, arguments=(), *, check=False, require_runtime=False, capture=False):
    path=Path(config).resolve()
    data=load_pipeline_config(path)
    if require_runtime and not check:
        raise ConfigurationError('require_runtime is only valid with check')
    command=[sys.executable,'-m',data['pipeline'],'--config',str(path)]
    if check: command.append('--check')
    if require_runtime: command.append('--require-runtime')
    command.extend(arguments)
    # Do not reinterpret native buffers, stage boundaries or model settings.
    return subprocess.run(command,text=True,capture_output=capture)
