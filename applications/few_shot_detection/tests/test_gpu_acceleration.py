"""GPU ROI parity and validation using prebuilt PyTorch CUDA operators."""
import numpy as np
import pytest
import torch

from dinov3_cases.platforms.nvidia.geometry import prepare_rois
from dinov3_cases.platforms.nvidia.cpu_roi import prepare_rois as cpu_prepare_rois

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')


@pytest.mark.parametrize('shape', [(20, 35), (35, 20), (1, 80), (80, 1), (1, 1), (77, 151)])
@pytest.mark.parametrize('size', [1, 16, 384])
def test_cuda_crops_preserve_center_padding_and_bilinear_pixels(shape, size):
    height, width = shape
    rng = np.random.default_rng(3)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    boxes = [[0, 0, width, height]]
    actual = prepare_rois(torch.from_numpy(pixels).cuda(), boxes, size)
    expected = cpu_prepare_rois(pixels, boxes, size)
    assert actual.device.type == 'cuda' and actual.dtype == torch.uint8
    assert actual.shape == expected.shape and actual.is_contiguous()
    assert abs(actual.cpu().numpy().astype(int) - expected.astype(int)).max() <= 1


def test_cuda_roi_preserves_offset_bounds_order_and_noncontiguous_input_on_stream():
    rng = np.random.default_rng(4)
    pixels = rng.integers(0, 256, (89, 111, 3), dtype=np.uint8)
    boxes = [[5, 7, 24, 48], [30, 20, 90, 37], [10, 10, 11, 11], [0, 0, 111, 89]]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Construct a noncontiguous HWC view on the same stream as ROI processing.
        rgb = torch.from_numpy(pixels).cuda().transpose(0, 1).contiguous().transpose(0, 1)
        assert not rgb.is_contiguous()
        actual = prepare_rois(rgb, boxes, 64)
    stream.synchronize()
    expected = cpu_prepare_rois(pixels, boxes, 64)
    assert abs(actual.cpu().numpy().astype(int) - expected.astype(int)).max() <= 1


@pytest.mark.parametrize('boxes', [
    [[0, 0, 50, 30]], [[-1, 0, 20, 20]], [[0, 0, 0, 20]],
    [[0, 0, 10, 31]], [[0, 0, 10.5, 20]], [[0, 0, float('nan'), 20]],
    [[0, 0, float('inf'), 20]], [[0, 0, 2**40, 20]], [[0, 0, 20]],
])
def test_cuda_roi_rejects_invalid_bounds(boxes):
    pixels = torch.zeros((30, 40, 3), dtype=torch.uint8, device='cuda')
    with pytest.raises(ValueError, match='ROI bounds'):
        prepare_rois(pixels, boxes)


@pytest.mark.parametrize('size', [0, -1, 2.5, True])
def test_cuda_roi_rejects_invalid_size_even_for_empty_batch(size):
    pixels = torch.zeros((30, 40, 3), dtype=torch.uint8, device='cuda')
    with pytest.raises(ValueError, match='ROI size'):
        prepare_rois(pixels, [], size)


def test_cuda_roi_empty_batch_and_invalid_image():
    pixels = torch.zeros((30, 40, 3), dtype=torch.uint8, device='cuda')
    for boxes in ([], np.empty((0, 4))):
        output = prepare_rois(pixels, boxes)
        assert output.shape == (0, 384, 384, 3)
        assert output.dtype == torch.uint8 and output.device == pixels.device
    for invalid in (pixels.cpu(), pixels.float(), pixels[:, :, :2], pixels[:0]):
        with pytest.raises(ValueError, match='CUDA uint8 HWC RGB'):
            prepare_rois(invalid, [])
