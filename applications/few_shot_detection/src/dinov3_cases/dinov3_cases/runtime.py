"""Platform-neutral application paths and model specifications."""
from pathlib import Path
import os

PROJECT = Path(os.environ.get("FEW_SHOT_HOME", Path(__file__).resolve().parents[3])).resolve()
ARTIFACTS = PROJECT / 'artifacts'
CASES_DIR = PROJECT / 'data/gallery'
MODEL_ID = 'zhanganyi88/Cat2Real-DINOv3-384'
GROUNDING_MODEL_ID = 'facebook/sam3'
SAM3_REVISION = '3c879f39826c281e95690f02c7821c4de09afae7'
INPUT_SIZE = 384
