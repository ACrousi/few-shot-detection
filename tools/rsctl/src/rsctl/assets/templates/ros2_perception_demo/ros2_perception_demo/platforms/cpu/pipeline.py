"""Complete CPU mean-RGB demonstration. No AI model or shared inference backend."""
import argparse,json
from pathlib import Path
import numpy as np
from robotic_inference.configuration import read_yaml

class Pipeline:
    def __init__(self,config_path):
        self.settings=read_yaml(config_path)
        if self.settings.get('platform')!='cpu' or self.settings.get('pipeline')!='ros2_perception_demo.platforms.cpu.pipeline':
            raise ValueError('CPU example requires its own platform configuration')
        self.closed=False
    def process(self,image):
        if self.closed: raise RuntimeError('Pipeline is closed')
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or not image.size:
            raise ValueError('Expected nonempty uint8 HWC RGB')
        return dict(mode='MOCK',mean_rgb=image.mean(axis=(0,1)).tolist(),shape=list(image.shape))
    def close(self): self.closed=True
    def __enter__(self): return self
    def __exit__(self,*exc): self.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--require-runtime',action='store_true')
    args=parser.parse_args()
    with Pipeline(args.config) as pipeline:
        if args.check:
            print(json.dumps(dict(platform='cpu',mode='MOCK',runtime='numpy',inference_verified=False)))
        else:
            print(json.dumps(pipeline.process(np.full((8,8,3),128,np.uint8))))

if __name__=='__main__': main()
