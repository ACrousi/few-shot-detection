"""Portable numeric tools and platform-owned selection/lifecycle."""
from pathlib import Path
import subprocess,sys
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from dinov3_cases.platforms import load_pipeline
from dinov3_cases.platforms.nvidia.cpu_roi import prepare_rois, normalize
from dinov3_cases.platforms.nvidia.preprocessing import prepare_cat2real_image
from dinov3_cases.runtime import PROJECT

@pytest.mark.parametrize('platform',['intel','qualcomm'])
def test_unsupported_platform_rejected_before_vendor_imports(platform):
    code="""
import sys
from dinov3_cases.platforms import load_pipeline
try:
    load_pipeline(sys.argv[1])
except ValueError as error:
    assert 'not implemented' in str(error)
else:
    raise AssertionError('unsupported platform accepted')
assert not any(name in sys.modules for name in ('torch','tensorrt','rclpy'))
"""
    subprocess.run([sys.executable,'-c',code,str(PROJECT/'config/platforms'/f'{platform}.yaml')],check=True)

@pytest.mark.parametrize('shape', [(20,35), (35,20), (1,80), (80,1)])
def test_cpu_roi_matches_reference(shape):
    rng=np.random.default_rng(12); h,w=shape
    rgb=rng.integers(0,256,(h+4,w+6,3),dtype=np.uint8)
    boxes=[[3,2,w+3,h+2]]
    actual=prepare_rois(rgb,boxes)
    expected=np.array(prepare_cat2real_image(Image.fromarray(rgb).crop(boxes[0])))
    np.testing.assert_array_equal(actual[0],expected)
    assert prepare_rois(rgb,[]).shape == (0,384,384,3)

@pytest.mark.parametrize('box', [[-1,0,2,2],[0,0,0,1],[0,0,30,30],[0,0,1.5,2],[0,0,float('nan'),2]])
def test_cpu_roi_rejects_invalid_bounds(box):
    with pytest.raises(ValueError,match='bounds'): prepare_rois(np.zeros((20,20,3),np.uint8),[box])

def test_cpu_normalization_contract():
    p=SimpleNamespace(do_rescale=True,rescale_factor=1/255,do_normalize=True,
                      image_mean=[.4,.5,.6],image_std=[.2,.3,.4])
    crops=np.full((2,3,4,3),255,np.uint8)
    out=normalize(crops,p)
    assert out.dtype==np.float32 and out.shape==(2,3,3,4) and out.flags.c_contiguous
    np.testing.assert_allclose(out[0,:,0,0],(1-np.array(p.image_mean))/p.image_std,rtol=1e-6)


def test_selection_is_relative_to_application_home(monkeypatch):
    monkeypatch.chdir('/')
    with load_pipeline('config/platforms/nvidia.yaml') as pipeline:
        assert pipeline.config['platform']=='nvidia'
        assert pipeline.encoder is None and pipeline.detector is None
    with pytest.raises(RuntimeError,match='closed'): pipeline.load_models()

def test_runtime_has_no_vendor_defaults():
    import dinov3_cases.runtime as runtime
    assert not hasattr(runtime,'DINO_ENGINE')
    assert not hasattr(runtime,'GROUNDING_ENGINE')

def test_partial_model_load_failure_cleans_encoder(monkeypatch):
    from unittest.mock import Mock
    monkeypatch.setattr('dinov3_cases.platforms.nvidia.encoder.Encoder',lambda *a,**k:Mock(device='cuda'))
    def fail(*args,**kwargs): raise RuntimeError('worker failed')
    monkeypatch.setattr('dinov3_cases.platforms.nvidia.sam3_worker.SharedMemoryGroundingDetector',fail)
    pipeline=load_pipeline('config/platforms/nvidia.yaml')
    with pytest.raises(RuntimeError,match='worker failed'): pipeline.load_models()
    assert pipeline.closed and pipeline.encoder is None and pipeline.detector is None

def test_cpu_roi_no_vendor_imports():
    subprocess.run([sys.executable,'-c',"from dinov3_cases.platforms.nvidia.cpu_roi import prepare_rois; import sys; assert 'torch' not in sys.modules"],check=True)

@pytest.mark.parametrize('section,key,value', [('execution','timeout',float('nan')),('execution','timeout',0),('operations','plane','invalid'),('operations','roi','auto')])
def test_platform_owns_validation(tmp_path,section,key,value):
    import yaml
    from robotic_inference.configuration import read_yaml
    data=read_yaml(PROJECT/'config/platforms/nvidia.yaml')
    data[section][key]=value
    cfg=tmp_path/'nvidia.yaml';cfg.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError): load_pipeline(cfg)
