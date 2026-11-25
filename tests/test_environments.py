import json
import os
from pathlib import Path
import subprocess
import sys
import venv
import pytest
from robotic_inference.errors import ConfigurationError
from rsctl import environments as e
from rsctl.cli import main, load_profile, init_app


def test_clean_environment(monkeypatch):
    for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','PYTHONUSERBASE'):
        monkeypatch.setenv(key,'/unrelated/ros/or/venv')
    env=e.clean_environment()
    assert not any(k in env for k in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','PYTHONUSERBASE'))
    assert env['PYTHONNOUSERSITE']=='1'


def test_existing_environment_not_modified(tmp_path):
    sentinel=tmp_path/'keep'; sentinel.write_text('user data')
    with pytest.raises(ConfigurationError,match='overwrite'):
        e.create(tmp_path,'intel',tmp_path,mock=True)
    assert sentinel.read_text()=='user data'


def test_invalid_interpreter_and_recipe(tmp_path):
    with pytest.raises(ConfigurationError,match='SDK'):
        e.create(tmp_path/'env','intel',tmp_path/'missing')
    with pytest.raises(ConfigurationError,match='requirements'):
        e.create(tmp_path/'env','intel',tmp_path,requirements=tmp_path/'missing')
    assert not (tmp_path/'env').exists()


def test_profile_environment_recipes():
    for name in ('nvidia','intel','qualcomm'):
        assert load_profile(name)['python_environment']['system_site_packages'] is False


def make_managed(tmp_path):
    path=tmp_path/'env'
    venv.EnvBuilder(with_pip=False).create(path)
    e.save(path,{'version':1,'path':str(path),'state':'ready','platform':'intel',
                'mock':True,'python_version':list(sys.version_info[:2]),'applications':{}})
    return path


def test_real_venv_run_isolated_and_exit_code(tmp_path,monkeypatch):
    path=make_managed(tmp_path)
    _,data=e.managed(path)
    modules=tmp_path/'installed'; modules.mkdir()
    pkg=modules/'demo'; pkg.mkdir()
    (pkg/'__init__.py').touch()
    output=tmp_path/'result.json'
    (pkg/'__main__.py').write_text('import json,os,sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(json.dumps({"prefix":sys.prefix,"args":sys.argv[2:],"paths":sys.path}))\nraise SystemExit(7)\n')
    data['applications']['demo']={'python_paths':[str(modules)]}
    e.save(path,data)
    contamination=tmp_path/'poison'; contamination.mkdir()
    (contamination/'sitecustomize.py').write_text('raise RuntimeError("must not load inherited PYTHONPATH")')
    monkeypatch.setenv('PYTHONPATH',str(contamination))
    code=main(['run','demo','--venv',str(path),'--',str(output),'--custom','hello world'])
    assert code==7
    result=json.loads(output.read_text())
    assert result['prefix']==str(path)
    assert result['args']==['--custom','hello world']
    assert str(contamination) not in result['paths']


def test_unmanaged_moved_and_incomplete(tmp_path):
    with pytest.raises(ConfigurationError,match='not an rsctl'):
        e.managed(tmp_path)
    path=make_managed(tmp_path)
    _,data=e.managed(path)
    data['state']='creating'; e.save(path,data)
    with pytest.raises(ConfigurationError,match='incomplete'): e.managed(path)
    data['state']='ready'; data['path']='/old/location'; e.save(path,data)
    with pytest.raises(ConfigurationError,match='moved'): e.managed(path)


def test_one_app_per_environment(tmp_path):
    path=make_managed(tmp_path)
    _,data=e.managed(path); data['applications']={'existing':{}}; e.save(path,data)
    with pytest.raises(ConfigurationError,match='one application'):
        e.install('other',path,source=tmp_path)
    with pytest.raises(ConfigurationError,match='not installed'):
        e.run('other',path,[])


def test_generated_module_has_no_ros_imports(tmp_path):
    path=init_app('scheme_a_demo',tmp_path,'ros2_perception_demo')
    code='import runpy,sys; sys.argv=["scheme_a_demo","--config",sys.argv[1]]; runpy.run_module("scheme_a_demo",run_name="__main__"); assert "rclpy" not in sys.modules; assert "cv_bridge" not in sys.modules'
    result=subprocess.run([sys.executable,'-c',code,str(path/'config/default.yaml')],cwd=path,check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)['mode']=='MOCK'


def test_cli_environment_errors(tmp_path,capsys):
    assert main(['env','check','--path',str(tmp_path)])==2
    assert 'not an rsctl' in capsys.readouterr().err
    assert main(['install','demo','--venv',str(tmp_path),'--source',str(tmp_path)])==2
