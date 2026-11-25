from pathlib import Path
import json,sys
import pytest
import yaml
from robotic_inference import load_pipeline_config,run_pipeline
from robotic_inference.errors import ConfigurationError

@pytest.mark.parametrize('data',[{},[],{'platform':'bad','pipeline':'demo.main'},{'platform':'intel','pipeline':'bad;command'}])
def test_invalid_launch_config(tmp_path,data):
    p=tmp_path/'cfg.yaml';p.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigurationError): load_pipeline_config(p)

def test_platform_owns_config_and_execution(tmp_path,monkeypatch):
    pkg=tmp_path/'example';pkg.mkdir();(pkg/'__init__.py').touch()
    (pkg/'pipeline.py').write_text("import sys,json; from pathlib import Path; print(json.dumps(dict(args=sys.argv[1:],cwd=str(Path.cwd())))); raise SystemExit(7 if '--fail' in sys.argv else 0)")
    monkeypatch.setenv('PYTHONPATH',str(tmp_path))
    cfg=tmp_path/'cfg.yaml';cfg.write_text('platform: intel\npipeline: example.pipeline\ncustom_vendor_buffers: anything\n')
    data=load_pipeline_config(cfg)
    assert data['custom_vendor_buffers']=='anything'
    result=run_pipeline(cfg,['--custom','hello world'],check=True,require_runtime=True,capture=True)
    assert result.returncode==0
    args=json.loads(result.stdout)['args']
    assert args==['--config',str(cfg),'--check','--require-runtime','--custom','hello world']
    assert run_pipeline(cfg,['--fail'],capture=True).returncode==7

def test_no_vendor_imports():
    import subprocess
    subprocess.run([sys.executable,'-c',"import robotic_inference; import sys; assert not any(x in sys.modules for x in ('numpy','torch','openvino','rclpy'))"],check=True)
