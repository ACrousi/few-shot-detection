"""Select an application implementation; all execution stays inside that platform."""
from pathlib import Path
from robotic_inference.configuration import read_yaml
from dinov3_cases.runtime import PROJECT


def load_pipeline(path):
    path = Path(path)
    path = path if path.is_absolute() else PROJECT/path
    data = read_yaml(path)
    if data.get('platform') != 'nvidia':
        raise ValueError(f"{data.get('platform')}: few-shot pipeline is not implemented; no fallback")
    from .nvidia.pipeline import Pipeline
    return Pipeline(path, config=data)
