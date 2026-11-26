"""GPU ROI preparation using prebuilt PyTorch CUDA operators."""
import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def prepare_rois(rgb, boxes, size=384):
    """CUDA uint8 HWC RGB -> uint8 NHWC centered white square crops.

    Boxes use integer, exclusive right/bottom bounds. Resize uses half-pixel
    bilinear sampling without antialiasing, matching the CPU OpenCV pipeline
    within one uint8 intensity level.
    """
    if (rgb.device.type != 'cuda' or rgb.dtype != torch.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1):
        raise ValueError('Expected CUDA uint8 HWC RGB')
    if type(size) is not int or size < 1:
        raise ValueError('ROI size must be a positive integer')
    bounds = np.asarray(boxes)
    if bounds.size == 0:
        return torch.empty((0, size, size, 3), dtype=torch.uint8, device=rgb.device)
    if (bounds.ndim != 2 or bounds.shape[1] != 4 or bounds.dtype.kind not in 'iuf'
            or not np.isfinite(bounds).all() or np.any(bounds != np.floor(bounds))
            or np.any(bounds[:, :2] < 0) or np.any(bounds[:, 2:] <= bounds[:, :2])
            or np.any(bounds[:, 2] > rgb.shape[1]) or np.any(bounds[:, 3] > rgb.shape[0])):
        raise ValueError('Invalid ROI bounds')
    output = torch.empty((len(bounds), size, size, 3), dtype=torch.uint8, device=rgb.device)
    for index, (x0, y0, x1, y1) in enumerate(bounds.astype(np.int64)):
        width, height = x1 - x0, y1 - y0
        side = max(width, height)
        left, top = (side - width) // 2, (side - height) // 2
        crop = rgb[y0:y1, x0:x1].permute(2, 0, 1).unsqueeze(0).float()
        square = F.pad(crop, (left, side - width - left, top, side - height - top), value=255)
        resized = F.interpolate(square, size=(size, size), mode='bilinear',
                                align_corners=False, antialias=False)
        output[index].copy_(resized[0].round_().clamp_(0, 255).to(torch.uint8).permute(1, 2, 0))
    return output
