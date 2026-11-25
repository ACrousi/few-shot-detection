"""Complete NVIDIA pipeline. No shared backend, tensor schema or stage factory."""
from pathlib import Path
import math
import sys
from robotic_inference.configuration import read_yaml


class Pipeline:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config = read_yaml(self.config_path)
        p = self.config
        if set(p) != {'platform', 'pipeline', 'models', 'operations', 'execution'}:
            raise ValueError('NVIDIA config requires platform, pipeline, models, operations, execution')
        if p['platform'] != 'nvidia' or p['pipeline'] != 'dinov3_cases.platforms.nvidia.pipeline':
            raise ValueError('Incorrect NVIDIA pipeline selection')
        if not isinstance(p['models'], dict) or set(p['models']) != {'sam3', 'cat2real'}:
            raise ValueError('NVIDIA models must contain sam3 and cat2real')
        for name, model in p['models'].items():
            if not isinstance(model, dict) or set(model) != {'artifact'} or not isinstance(model['artifact'], str) or not model['artifact']:
                raise ValueError(f'{name} requires an artifact path')
            model['artifact'] = str((self.config_path.parent/model['artifact']).resolve())
        operations = p['operations']
        if (not isinstance(operations, dict) or set(operations) - {'roi', 'plane'}
                or operations.get('roi') not in ('cpu', 'cuda')
                or operations.get('plane', 'cpu') not in ('cpu', 'cuda')):
            raise ValueError('NVIDIA operations.roi must be cpu or cuda')
        # Older deployment files can still contain the RANSAC backend setting.
        operations.pop('plane', None)
        e=p['execution']
        if not isinstance(e, dict) or set(e)-{'sam3','timeout','python'} or e.get('sam3') not in ('subprocess','inprocess'):
            raise ValueError('NVIDIA execution.sam3 must be subprocess or inprocess')
        timeout=e.get('timeout',60.)
        if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
            raise ValueError('NVIDIA timeout must be finite positive seconds')
        if 'python' in e and (not isinstance(e['python'],str) or not e['python']):
            raise ValueError('execution.python must be an executable path')
        self.encoder=self.detector=self.gallery=None
        self.closed=False
        self.settings=None

    def load_models(self):
        if self.closed:
            raise RuntimeError('Pipeline is closed')
        if self.encoder is not None and self.detector is not None:
            return
        try:
            from .encoder import Encoder
            from .recognition import ObjectGallery
            self.encoder=Encoder(self.config['models']['cat2real']['artifact'],roi_backend=self.config['operations']['roi'])
            e=self.config['execution']
            if e['sam3']=='subprocess':
                from .transport import SharedMemoryGroundingDetector
                self.detector=SharedMemoryGroundingDetector(self.config['models']['sam3']['artifact'],self.encoder.device,
                    python=e.get('python',sys.executable),timeout=e.get('timeout',60.))
            else:
                from .sam3 import Sam3Detector
                self.detector=Sam3Detector(self.config['models']['sam3']['artifact'],self.encoder.device)
            self.gallery=ObjectGallery()
        except BaseException:
            self.close()
            raise

    def configure_geometry(self, param):
        from dinov3_cases.geometry.depth_bbox import ProjectionRayCache
        from dinov3_cases.geometry.cloud_filter import CloudFilter
        self.depth_ray_cache=ProjectionRayCache()
        self.cloud_filter=CloudFilter(min_coverage=param('bbox3d_min_cloud_coverage'),
            min_coverage_ratio=param('bbox3d_min_coverage_ratio'),max_depth_jump=param('bbox3d_max_depth_jump'),
            confirmations=param('bbox3d_depth_confirmations'))

    def run_live(self, node):
        if self.config['execution']['sam3'] != 'subprocess':
            raise ValueError('NVIDIA live frame leases currently require subprocess transport')
        from .live import run
        run(node,self)

    def estimate_bboxes3d(self,node,detections,inputs):
        from .depth import estimate_bboxes3d
        return estimate_bboxes3d(node,self,detections,inputs)

    def configure(self,settings):
        self.settings=dict(settings)
        return self

    def infer(self,image):
        self.load_models()
        if self.settings is None:
            raise RuntimeError('Configure application settings before inference')
        from .recognition import recognize_objects
        from dinov3_cases.runtime import INPUT_SIZE
        image=image.convert('RGB')
        if image.width*image.height>25_000_000:
            raise ValueError('Input exceeds 25 megapixels')
        s=self.settings
        self.gallery.refresh(s['gallery'],self.encoder,INPUT_SIZE,16)
        proposals=self.detector.detect(image,s['prompt'],s['box_threshold'],s['max_detections'])
        return recognize_objects(image,proposals,self.encoder,self.gallery,INPUT_SIZE,16,s['recognition_threshold'])

    def validate(self,require_runtime=False):
        for name,model in self.config['models'].items():
            p=Path(model['artifact'])
            if not p.is_file() or not p.with_suffix('.json').is_file():
                raise ValueError(f'{name}: missing model artifact or metadata: {p}')
        if require_runtime:
            self.load_models()
            if 'cuda' in self.config['operations'].values():
                from .geometry import library
                library()
        return dict(platform='nvidia',artifacts='present',models_loaded=require_runtime,inference_verified=False)

    def reference_preview(self,image,stride=None):
        from .preprocessing import prepare_cat2real_image
        from dinov3_cases.runtime import INPUT_SIZE
        reference=prepare_cat2real_image(image,INPUT_SIZE)
        if stride is not None:
            from .patches import reference_grid_image
            reference=reference_grid_image(reference,stride)
        return reference

    def status(self):
        return dict(platform='nvidia',grounding_backend='trt',inference_backend='trt',
            grounding_execution=self.config['execution']['sam3'],grounding_engine_path=self.config['models']['sam3']['artifact'],
            platform_operations=dict(self.config['operations']))

    def close(self):
        self.closed=True
        try:
            close=getattr(self.detector,'close',None)
            if close: close()
        finally:
            self.detector=self.encoder=self.gallery=None

    def __enter__(self): return self
    def __exit__(self,*exc): self.close()


def main():
    import argparse,json
    from PIL import Image
    parser=argparse.ArgumentParser(description='NVIDIA-owned few-shot pipeline')
    parser.add_argument('--config',required=True)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--require-runtime',action='store_true')
    parser.add_argument('--image')
    parser.add_argument('--gallery')
    parser.add_argument('--prompt',default='box, can.')
    args=parser.parse_args()
    try:
        with Pipeline(args.config) as pipeline:
            if args.check:
                print(json.dumps(pipeline.validate(args.require_runtime),indent=2))
            else:
                if not args.image or not args.gallery: parser.error('--image and --gallery are required')
                pipeline.configure(dict(gallery=Path(args.gallery),prompt=args.prompt,box_threshold=.6,recognition_threshold=.6,max_detections=20))
                with Image.open(args.image) as image:
                    print(json.dumps(pipeline.infer(image),indent=2))
    except (ValueError,RuntimeError,OSError) as error:
        parser.exit(2,f'{error}\n')

if __name__=='__main__': main()
