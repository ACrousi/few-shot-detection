from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from dinov3_cases.platforms import load_pipeline
from dinov3_cases.runtime import PROJECT

def main():
    provider=load_pipeline(PROJECT/'config/platforms/nvidia.yaml')
    try:
        report=provider.validate(require_runtime=True)
        encoder=provider.encoder
        image=Image.open(PROJECT/'artifacts/validation/person.jpg').convert('RGB')
        detector=provider.detector
        boxes=detector.detect(image,'person.',.5,20)
        assert boxes, 'SAM3 must detect the validation person'
        coords=[p['bbox'] for p in boxes]
        gpu=encoder.encode_regions(image,coords)
        encoder.roi_backend='cpu'
        cpu=encoder.encode_regions(image,coords)
        cosines=[]; pixel_errors=[]
        for g,c in zip(gpu,cpu):
            cosine=float(torch.dot(g[2],c[2])); cosines.append(cosine)
            error=int(np.abs(np.array(g[0]).astype(int)-np.array(c[0]).astype(int)).max());pixel_errors.append(error)
            assert error<=1, error
            assert cosine>.999, cosine
        from dinov3_cases.geometry.depth_bbox import attach_bboxes3d
        from sensor_msgs.msg import CameraInfo

        info = CameraInfo(width=160, height=120,
                          k=[200., 0., 80., 0., 200., 60., 0., 0., 1.])
        items = [dict(bbox=[35, 40, 125, 80])]
        attach_bboxes3d(items, np.ones((120, 160)), info, quantile=0.)
        assert items[0]["bbox3d"]["method"] == "central_depth_component_plane_rectangle"
        assert items[0]["bbox3d"]["plane_backend"] == "open3d_cpu"
        assert items[0]["bbox3d"]["fit_status"] == "matched"
        np.testing.assert_allclose(items[0]["bbox3d"]["size"], [.445, .195, 0.])
        report.update(detections=len(boxes), roi_descriptor_cosines=cosines, roi_max_pixel_errors=pixel_errors,
                      inference_verified=True, geometry_verified=True)
        print(json.dumps(report,indent=2))
    finally:
        provider.close()

if __name__ == "__main__":
    main()
