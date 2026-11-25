"""Portable Cat2Real ROI preparation. No Torch, CUDA, ROS or runtime imports."""
import cv2
import numpy as np


def prepare_rois(rgb, boxes, size=384):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape) < 1:
        raise ValueError("Expected uint8 HWC RGB")
    if type(size) is not int or size < 1:
        raise ValueError("ROI size must be a positive integer")
    bounds = np.asarray(boxes)
    if bounds.size == 0:
        return np.empty((0, size, size, 3), dtype=np.uint8)
    if (bounds.ndim != 2 or bounds.shape[1] != 4 or bounds.dtype.kind not in 'iuf'
            or not np.isfinite(bounds).all() or np.any(bounds != np.floor(bounds))
            or np.any(bounds[:, :2] < 0) or np.any(bounds[:, 2:] <= bounds[:, :2])
            or np.any(bounds[:, 2] > rgb.shape[1]) or np.any(bounds[:, 3] > rgb.shape[0])):
        raise ValueError("Invalid ROI bounds")
    result = []
    for x0, y0, x1, y1 in bounds.astype(np.int64):
        w, h = x1-x0, y1-y0
        side = max(w, h)
        square = np.full((side, side, 3), 255, dtype=np.uint8)
        left, top = (side-w)//2, (side-h)//2
        square[top:top+h, left:left+w] = rgb[y0:y1, x0:x1]
        result.append(cv2.resize(square, (size, size), interpolation=cv2.INTER_LINEAR))
    return np.stack(result)


def normalize(crops, processor):
    """uint8 NHWC -> float32 NCHW; same model processor contract as GPU."""
    crops = np.asarray(crops)
    if crops.dtype != np.uint8 or crops.ndim != 4 or crops.shape[-1] != 3:
        raise ValueError("Expected uint8 NHWC crops")
    pixels = np.ascontiguousarray(crops.transpose(0, 3, 1, 2), dtype=np.float32)
    if processor.do_rescale:
        pixels *= np.float32(processor.rescale_factor)
    if processor.do_normalize:
        mean = np.asarray(processor.image_mean, dtype=np.float32).reshape(1,3,1,1)
        std = np.asarray(processor.image_std, dtype=np.float32).reshape(1,3,1,1)
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Invalid normalization parameters")
        pixels -= mean
        pixels /= std
    return pixels
