"""Application settings and direct platform selection; no model execution here."""
from pathlib import Path
from robotic_inference.configuration import read_yaml
from dinov3_cases.platforms import load_pipeline


def load_settings(path):
    values=read_yaml(path)
    required={'platform_config','prompt','box_threshold','recognition_threshold','max_detections','gallery'}
    if set(values)!=required: raise ValueError(f'Expected configuration keys: {sorted(required)}')
    if not isinstance(values['platform_config'],str) or not values['platform_config']:
        raise ValueError('platform_config must be a path')
    values['platform_config']=(Path(path).resolve().parent/values['platform_config']).resolve()
    with load_pipeline(values['platform_config']): pass
    from dinov3_cases.recognition.settings import validate_settings
    values['prompt']=validate_settings(values['prompt'],values['box_threshold'],values['recognition_threshold'],0)
    limit=values['max_detections']
    if type(limit) is not int or limit<=0: raise ValueError('max_detections must be a positive integer')
    values['gallery']=(Path(path).resolve().parent/values['gallery']).resolve()
    if not values['gallery'].is_dir(): raise ValueError(f"Gallery directory missing: {values['gallery']}")
    return values


def open_pipeline(settings):
    return load_pipeline(settings['platform_config']).configure(settings)
