from pathlib import Path
import numpy as np
from ros2_perception_demo.platforms.cpu.pipeline import Pipeline

def test_mock_pipeline():
    with Pipeline(Path(__file__).parents[1]/'config/default.yaml') as pipeline:
        result = pipeline.process(np.zeros((4,4,3),dtype=np.uint8))
    assert result['mode'] == 'MOCK'
    assert result['mean_rgb'] == [0,0,0]
