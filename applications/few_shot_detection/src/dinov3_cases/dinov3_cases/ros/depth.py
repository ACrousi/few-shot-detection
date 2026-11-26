"""ROS RGB-D conversion and CPU RANSAC bounding-box estimation."""
from cv_bridge import CvBridgeError
from dinov3_cases.geometry.depth_bbox import attach_bboxes3d

def estimate_bboxes3d(bridge, param, pipeline, detections, inputs):
    if inputs is None:
        for item in detections:
            item["bbox3d_status"] = "missing_synchronized_registered_depth"
        return
    depth_msg, info = inputs
    try:
        depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth_msg.encoding == "16UC1":
            depth_scale = 0.001
        elif depth_msg.encoding == "32FC1":
            depth_scale = 1.0
        else:
            raise ValueError(f"unsupported_depth_encoding:{depth_msg.encoding}")
        attach_bboxes3d(
            detections, depth, info,
            depth_scale=depth_scale, ray_cache=pipeline.depth_ray_cache,
            min_depth=param("pose_min_depth"), max_depth=param("pose_max_depth"),
            depth_band=param("bbox3d_depth_band"),
            min_points=param("bbox3d_min_points"),
            quantile=param("bbox3d_quantile"),
            cloud_filter=pipeline.cloud_filter,
            cloud_frame=(depth_msg.header.frame_id, depth_msg.header.stamp.sec,
                         depth_msg.header.stamp.nanosec),
        )
    except (ValueError, CvBridgeError) as error:
        for item in detections:
            item["bbox3d_status"] = str(error)
