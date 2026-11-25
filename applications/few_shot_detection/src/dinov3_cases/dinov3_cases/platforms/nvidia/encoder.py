"""Cat2Real pooler descriptor and spatial features from one TRT forward."""
import numpy as np
import torch
from transformers import AutoImageProcessor

from dinov3_cases.platforms.nvidia.patches import preprocess_image, patch_features
from dinov3_cases.platforms.nvidia.preprocessing import prepare_cat2real_image
from dinov3_cases.runtime import MODEL_ID, INPUT_SIZE
from dinov3_cases.platforms.nvidia.artifacts import DINO_ENGINE, DINO_MODEL
from dinov3_cases.platforms.nvidia.trt_backend import TRTBackbone


class Encoder:
    def __init__(self, engine_path=DINO_ENGINE, *, roi_backend="cuda"):
        if roi_backend not in ("cuda", "cpu"):
            raise ValueError("roi_backend must be cuda or cpu")
        self.roi_backend = roi_backend
        self.model = TRTBackbone(engine_path, MODEL_ID)
        self.device = str(self.model.device)
        self.description = self.model.description
        self.processor = AutoImageProcessor.from_pretrained(str(DINO_MODEL), local_files_only=True)

    def prepare_image(self, image, max_side, stride):
        return prepare_cat2real_image(image, INPUT_SIZE)

    @torch.inference_mode()
    def encode_with_descriptor(self, image):
        pixels = torch.from_numpy(np.array(image.convert('RGB'), dtype=np.uint8)).permute(2, 0, 1).to(self.device)
        outputs = self.model(pixel_values=preprocess_image(pixels, self.processor))
        features = patch_features(outputs, self.model, image.height, image.width)
        features = features[0].reshape(-1, features.shape[-1])
        descriptor = outputs.pooler_output[0].float()
        if not torch.isfinite(features).all() or not torch.isfinite(descriptor).all():
            raise ValueError('Non-finite Cat2Real image features')
        return features, descriptor

    @torch.inference_mode()
    def encode_regions(self, image, boxes):
        """Prepare ROIs on the configured device and infer current microbatches."""
        from PIL import Image
        if not boxes:
            return []
        cpu_roi = getattr(self, "roi_backend", "cuda") == "cpu"
        if cpu_roi:
            from dinov3_cases.platforms.nvidia.cpu_roi import prepare_rois as cpu_prepare, normalize
            prepared = cpu_prepare(np.asarray(image.convert('RGB')), boxes, INPUT_SIZE)
        else:
            from dinov3_cases.platforms.nvidia.geometry import prepare_rois
            rgb = torch.from_numpy(np.array(image.convert('RGB'), dtype=np.uint8)).to(self.device)
            prepared = prepare_rois(rgb, boxes, INPUT_SIZE)
        maximum = self.model.maximum[0]
        rows = []
        for start in range(0, len(boxes), maximum):
            crops = prepared[start:start+maximum]
            if cpu_roi:
                pixels = torch.from_numpy(normalize(crops, self.processor)).to(self.device)
            else:
                pixels = crops.permute(0,3,1,2).contiguous().float()
                if self.processor.do_rescale:
                    pixels.mul_(self.processor.rescale_factor)
                if self.processor.do_normalize:
                    mean = pixels.new_tensor(self.processor.image_mean).view(1,3,1,1)
                    std = pixels.new_tensor(self.processor.image_std).view(1,3,1,1)
                    pixels.sub_(mean).div_(std)
            outputs = self.model(pixel_values=pixels)
            features = patch_features(outputs, self.model, INPUT_SIZE, INPUT_SIZE).flatten(1,2)
            descriptors = outputs.pooler_output.float()
            valid = (torch.isfinite(features).all() & torch.isfinite(descriptors).all()
                     & (descriptors.norm(dim=1) > 1e-8).all())
            if not bool(valid):
                raise ValueError('Non-finite or empty Cat2Real batch features')
            descriptors = torch.nn.functional.normalize(descriptors, dim=1)
            previews = crops if cpu_roi else crops.cpu().numpy()  # One batch transfer; these are actual model pixels.
            for offset, preview in enumerate(previews):
                box = boxes[start+offset]
                width,height = box[2]-box[0],box[3]-box[1]
                side=max(width,height);left,top=(side-width)//2,(side-height)//2
                scale=INPUT_SIZE/side
                x0,y0=round(left*scale),round(top*scale)
                x1,y1=round((left+width)*scale),round((top+height)*scale)
                diagnostic = Image.fromarray(preview)
                diagnostic.info['content_box']=(x0,y0,max(x0+1,min(INPUT_SIZE,x1)),max(y0+1,min(INPUT_SIZE,y1)))
                rows.append((diagnostic, features[offset], descriptors[offset]))
        return rows
