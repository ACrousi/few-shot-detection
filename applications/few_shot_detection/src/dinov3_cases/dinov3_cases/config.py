"""Startup parameter defaults and validation, before workers are started.

Live recognition settings are validated by recognition.settings.validate_settings as well.
Keep deployment overrides in config/dinov3_cases.yaml.
"""

import numpy as np

from dinov3_cases.runtime import CASES_DIR
from dinov3_cases.tracking.kcf import check_kcf, validate_tracker_settings
from dinov3_cases.tracking.async_tracker import validate_missed_seconds
from dinov3_cases.geometry.async_bbox3d import validate_bbox3d_max_age
from dinov3_cases.recognition.settings import DEFAULT_PROMPT, validate_settings, validate_bbox_limits, validate_containment_threshold


# These ROS parameters accept both integer and floating-point seconds.
SECONDS_PARAMETERS = frozenset({
    "grounding_detection_interval", "bbox_smoothing_seconds",
    "tracker_max_missed_seconds", "pose_sync_tolerance",
    "bbox3d_max_age",
})


def default_parameters():
    """Return a fresh set of ROS parameter defaults."""
    return dict(
        platform_config="config/platforms/nvidia.yaml",
        cases_dir=str(CASES_DIR),
        grounding_prompt=DEFAULT_PROMPT,
        grounding_box_threshold=0.3,
        grounding_containment_threshold=0.95,
        grounding_max_detections=20,
        grounding_detection_interval=0.0,
        bbox_smoothing_seconds=0.1,
        opencv_num_threads=0,
        kcf_detect_threshold=0.5,
        tracker_width=640,
        tracker_height=384,
        tracker_max_missed_seconds=1.0,
        grounding_min_bbox_width=0,
        grounding_min_bbox_height=0,
        grounding_min_bbox_area=0,
        sam3_roi_width=0,
        sam3_roi_height=0,
        sam3_roi_center_x=-1,
        sam3_roi_center_y=-1,
        recognition_threshold=0.75,
        recognition_margin=0.05,  # ROS/API compatibility only; never passed to classification.
        image_topic="/camera/color/image_raw",
        annotated_image_topic="/medicine_recognition/image_annotated",
        tag_topic="/tag_detections",
        pose_enabled=True,
        depth_topic="/camera/depth/image_raw",
        depth_camera_info_topic="/camera/depth/camera_info",
        pose_sync_tolerance=0.067,  # Seconds; independent of video_fps.
        pose_min_depth=0.1,
        pose_max_depth=5.0,
        bbox3d_max_age=1.5,  # Seconds from source capture.
        bbox3d_depth_band=0.08,
        bbox3d_min_points=80,
        bbox3d_quantile=0.02,
        bbox3d_min_cloud_coverage=0.15,
        bbox3d_min_coverage_ratio=0.35,
        bbox3d_max_depth_jump=0.10,
        bbox3d_depth_confirmations=3,
        port=8090,
        host="127.0.0.1",
        video_fps=30,
        bbox_visualization_mode="3d",
    )


def validate_parameters(param):
    """Validate startup values and return normalized recognition settings."""
    if param("bbox_visualization_mode") not in ("2d", "3d"):
        raise ValueError("bbox_visualization_mode must be 2d or 3d.")
    opencv_threads = param("opencv_num_threads")
    if isinstance(opencv_threads, bool) or not isinstance(opencv_threads, int) or opencv_threads < 0:
        raise ValueError("opencv_num_threads must be an integer >= 0 (0 keeps the OpenCV default).")
    settings = dict(
        prompt=param("grounding_prompt"),
        box_threshold=param("grounding_box_threshold"),
        recognition_threshold=param("recognition_threshold"),
        margin=param("recognition_margin"),
        detection_interval=param("grounding_detection_interval"),
    )
    settings["prompt"] = validate_settings(**settings)
    validate_missed_seconds(param("tracker_max_missed_seconds"))
    check_kcf()
    validate_tracker_settings(param("tracker_width"), param("tracker_height"), param("kcf_detect_threshold"))
    validate_containment_threshold(param("grounding_containment_threshold"))
    if not 1 <= param("grounding_max_detections") <= 100:
        raise ValueError("grounding_max_detections must be between 1 and 100.")
    validate_bbox_limits(
        param("grounding_min_bbox_width"),
        param("grounding_min_bbox_height"),
        param("grounding_min_bbox_area"),
    )
    fps = param("video_fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or not 1 <= fps <= 30:
        raise ValueError("video_fps must be between 1 and 30.")
    tolerance = param("pose_sync_tolerance")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("pose_sync_tolerance must be finite seconds >= 0.")
    smoothing = param("bbox_smoothing_seconds")
    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)) or not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("bbox_smoothing_seconds must be finite seconds >= 0.")
    for name in ("pose_min_depth", "pose_max_depth", "bbox3d_depth_band"):
        if not np.isfinite(param(name)) or param(name) <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not 0 <= param("bbox3d_quantile") < 0.5:
        raise ValueError("bbox3d_quantile must be in [0, 0.5).")
    if (not isinstance(param("bbox3d_min_points"), int)
            or param("bbox3d_min_points") < 5):
        raise ValueError("bbox3d_min_points must be an integer >= 5.")
    if param("pose_min_depth") >= param("pose_max_depth"):
        raise ValueError("pose_min_depth must be less than pose_max_depth.")
    validate_bbox3d_max_age(param("bbox3d_max_age"))
    return settings
