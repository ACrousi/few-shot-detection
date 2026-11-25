"""Read launch settings; the selected platform validates its own model configuration."""
from pathlib import Path
import re
import yaml
from .errors import ConfigurationError


def read_yaml(path):
    path=Path(path).resolve()
    try:
        data=yaml.safe_load(path.read_text())
    except (OSError,yaml.YAMLError) as error:
        raise ConfigurationError(f'cannot read YAML {path}: {error}') from error
    if not isinstance(data,dict): raise ConfigurationError(f'{path}: expected a YAML mapping')
    return data


def load_pipeline_config(path):
    data=read_yaml(path)
    if data.get('platform') not in ('nvidia','intel','qualcomm','cpu'):
        raise ConfigurationError('platform must be nvidia, intel, qualcomm or cpu')
    module=data.get('pipeline')
    if not isinstance(module,str) or not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+',module):
        raise ConfigurationError('pipeline must be an explicit Python module name')
    return data
