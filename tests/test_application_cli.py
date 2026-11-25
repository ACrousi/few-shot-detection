import importlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import yaml
from robotic_inference import load_pipeline_config
from robotic_inference.errors import ConfigurationError
from rsctl.cli import ASSETS, init_app, load_profile, doctor, build_plan, main
ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ASSETS/'templates/ros2_perception_demo/config/default.yaml'

def test_generated_standalone_package(tmp_path):
    path = init_app('sample_vision',tmp_path,'ros2_perception_demo')
    assert ET.parse(path/'package.xml').getroot().findtext('name')=='sample_vision'
    assert (path/'resource/sample_vision').is_file()
    assert 'ros2_perception_demo' not in (path/'setup.py').read_text()
    sys.path.insert(0,str(path))
    try:
        pipeline = importlib.import_module('sample_vision.platforms.cpu.pipeline').Pipeline
        with pipeline(path/'config/default.yaml') as p:
            assert p.process(np.zeros((5,6,3),dtype=np.uint8))['mean_rgb']==[0,0,0]
            with pytest.raises(ValueError): p.process(np.zeros((2,2)))
        subprocess.run([sys.executable,'-m','pytest',str(path/'tests'),'-q'],cwd=path,check=True)
    finally: sys.path.remove(str(path))
    with pytest.raises(ConfigurationError,match='overwrite'): init_app('sample_vision',tmp_path,'ros2_perception_demo')

@pytest.mark.parametrize('name',['../bad','Upper','bad-name','class',''])
def test_invalid_package_name(name,tmp_path):
    with pytest.raises(ConfigurationError): init_app(name,tmp_path,'ros2_perception_demo')

@pytest.mark.parametrize('name',['nvidia','intel','qualcomm'])
def test_profiles_and_doctor(name):
    assert load_profile(name)['name']==name
    result=doctor(name)
    assert {r['status'] for r in result} <= {'passed','failed','not installed','not verifiable'}
    assert any(r['check']=='architecture' for r in result)
    plan=build_plan(name,'sample_vision',ROOT,None)
    assert 'docker build' in plan['command']
    assert plan['missing'] and not plan['execution_supported']

def test_cli(capsys, monkeypatch):
    # The template is not installed until generation; expose it for subprocess validation.
    monkeypatch.setenv('PYTHONPATH', str(DEFAULT.parent.parent))
    assert main(['validate','--config',str(DEFAULT)])==0
    assert 'Configuration valid' in capsys.readouterr().out
    assert main(['doctor','--platform','intel','--json']) in (0,1)
    assert isinstance(json.loads(capsys.readouterr().out),list)
    assert main(['build','--platform','intel','--app','demo','--dry-run'])==0
    assert 'DRY RUN' in capsys.readouterr().out
    assert main(['build','--platform','intel','--app','demo'])==2
    assert 'Real builds are not enabled' in capsys.readouterr().err
    assert main(['doctor','--platform','unknown'])==2
    assert main(['validate','--config','missing.yaml'])==2

@pytest.mark.parametrize('data',[[],{}, {'application':{'name':'x'},'models':{}}, {'application':{'name':'x'},'models':{'a':{'backend':'unknown'}}}, {'application':{'name':'x'},'models':{'a':{'backend':'mock'}},'platform':'bad'}])
def test_invalid_configuration(tmp_path,data):
    p=tmp_path/'bad.yaml'; p.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigurationError): load_pipeline_config(p)

def test_build_context_package_mismatch(tmp_path):
    app=tmp_path/'application'; app.mkdir()
    (app/'package.xml').write_text('<package><name>different</name></package>')
    with pytest.raises(ConfigurationError,match='must match'): build_plan('intel','demo',tmp_path,'image')
