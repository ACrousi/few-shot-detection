"""NVIDIA-only artifact defaults for model build and diagnostic tools."""
from pathlib import Path
import sys
from dinov3_cases.runtime import ARTIFACTS

DINO_ENGINE_SINGLE = ARTIFACTS / 'cat2real-dinov3-384/cat2real-fp32-trt.ts'
DINO_ENGINE_BATCH = ARTIFACTS / 'cat2real-dinov3-384/cat2real-fp32-batch8-trt.ts'
DINO_ENGINE = DINO_ENGINE_BATCH if DINO_ENGINE_BATCH.with_suffix('.json').is_file() else DINO_ENGINE_SINGLE
DINO_MODEL = ARTIFACTS / 'cat2real-dinov3-384/source'
SAM3_ASSETS = ARTIFACTS / 'sam3'
SAM3_MODEL = SAM3_ASSETS / 'source'
GROUNDING_ENGINE_SINGLE = SAM3_ASSETS / 'vision-fp16.ts'
GROUNDING_ENGINE_NATIVE = SAM3_ASSETS / 'vision-fp16-native.ts'
GROUNDING_ENGINE = GROUNDING_ENGINE_NATIVE if GROUNDING_ENGINE_NATIVE.with_suffix('.json').is_file() else GROUNDING_ENGINE_SINGLE
GROUNDING_PYTHON = Path(sys.executable)
