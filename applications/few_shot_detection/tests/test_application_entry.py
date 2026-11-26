"""Application configuration and standalone boundaries, independent of ROS."""
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from few_shot_detection.standalone import load_settings


def configuration(tmp_path, **overrides):
    (tmp_path / 'gallery').mkdir()
    from dinov3_cases.runtime import PROJECT
    values = dict(platform_config=str(PROJECT/'config/platforms/nvidia.yaml'), prompt='person', box_threshold=.5,
                  recognition_threshold=.6, max_detections=20, gallery='gallery')
    values.update(overrides)
    path = tmp_path / 'native.yaml'
    path.write_text(yaml.safe_dump(values))
    return path


def test_paths_are_relative_to_config_not_working_directory(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    monkeypatch.chdir('/')
    settings = load_settings(config)
    assert settings['gallery'] == tmp_path / 'gallery'
    assert settings['prompt'] == 'person.'


@pytest.mark.parametrize('overrides', [dict(runtime='mock'), dict(max_detections=True),
                                      dict(max_detections=0), dict(box_threshold=2),
                                      dict(gallery='missing'), dict(extra=True)])
def test_invalid_deployment_fails_without_runtime_fallback(tmp_path, overrides):
    with pytest.raises(ValueError):
        load_settings(configuration(tmp_path, **overrides))


def test_configuration_does_not_import_ros(tmp_path):
    config = configuration(tmp_path)
    subprocess.run([sys.executable, '-c',
                    'import sys; from few_shot_detection.standalone import load_settings; '
                    'load_settings(sys.argv[1]); '
                    'assert "rclpy" not in sys.modules; assert "cv_bridge" not in sys.modules',
                    str(config)], check=True, cwd=tmp_path)


def test_closed_pipeline_rejects_inference():
    from dinov3_cases.platforms import load_pipeline
    pipeline=load_pipeline('config/platforms/nvidia.yaml')
    pipeline.close(); pipeline.close()
    with pytest.raises(RuntimeError,match='closed'):
        pipeline.infer(None)


def test_single_image_entry_reads_platform_yaml_once(tmp_path, monkeypatch):
    from unittest.mock import patch
    from PIL import Image
    from robotic_inference.configuration import read_yaml
    from few_shot_detection.__main__ import main

    config = configuration(tmp_path)
    image = tmp_path / "input.png"
    Image.new("RGB", (4, 4)).save(image)
    monkeypatch.setattr(sys, "argv", ["infer", "--config", str(config), "--image", str(image)])
    with patch("dinov3_cases.platforms.read_yaml", wraps=read_yaml) as select_read, \
         patch("dinov3_cases.platforms.nvidia.pipeline.read_yaml", wraps=read_yaml) as platform_read, \
         patch("dinov3_cases.platforms.nvidia.pipeline.Pipeline.infer", return_value=[]):
        main()
    assert select_read.call_count + platform_read.call_count == 1
