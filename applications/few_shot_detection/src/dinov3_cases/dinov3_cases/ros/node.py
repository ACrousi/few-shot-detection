"""ROS input/output and selected platform pipeline lifecycle."""

import json
import threading
import time
from collections import deque
from pathlib import Path

import cv2
from PIL import Image
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image as ImageMsg, CameraInfo
from std_msgs.msg import Header, String
from werkzeug.serving import make_server

from dinov3_cases.timing import ResultRate, ModelTimings
from dinov3_cases.platforms import load_pipeline
from dinov3_cases.frame_pool import SharedFramePool
from dinov3_cases.tracking.identities import BBoxSmoother
from dinov3_cases.config import SECONDS_PARAMETERS, default_parameters, validate_parameters
from dinov3_cases.geometry.async_bbox3d import AsyncBBox3D
from vision_msgs.msg import Detection2DArray, Detection3DArray
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
from dinov3_cases.ros.depth import estimate_bboxes3d
from dinov3_cases.ros.messages import apriltag_message
from dinov3_cases.geometry.depth_sync import matched_inputs
from dinov3_cases.live_state import InferenceUnavailable
from dinov3_cases.inference.sam3_roi import DEFAULT_ROI, filter_roi, roi_bounds, validate_roi
from dinov3_cases.api.image_stream import ImageStream
from dinov3_cases.ros.messages import bboxes3d_message
from dinov3_cases.api.routes import create_app
from dinov3_cases.api.visualization import render_display_overlay
from dinov3_cases.ros.messages import recognition_message


class FewShotDetectionNode(Node):
    def __init__(self):
        super().__init__("dinov3_cases")
        self.readiness_lock = threading.Lock()
        self.readiness = dict(state="loading", message="Loading DINOv3…")
        self.readiness_pub = self.create_publisher(
            String, "/medicine_recognition/status",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.readiness_timer = self.create_timer(1.0, self.publish_readiness)
        self.publish_readiness()
        defaults = default_parameters()
        for k, v in defaults.items():
            if k in SECONDS_PARAMETERS:
                self.declare_parameter(k, v, ParameterDescriptor(
                    dynamic_typing=True, description="Duration in seconds; independent of video_fps."))
            else:
                self.declare_parameter(k, v)
        self.param = lambda k: self.get_parameter(k).value
        self.sam3_roi = validate_roi(*(self.param("sam3_roi_" + key)
                                     for key in ("width", "height", "center_x", "center_y")))
        self.grounding_settings = validate_parameters(self.param)
        self.platform_pipeline = load_pipeline(self.param("platform_config"))
        self.bbox_visualization_mode = self.param("bbox_visualization_mode")
        # Set the global OpenCV pool before starting inference/render workers.
        opencv_threads = self.param("opencv_num_threads")
        if opencv_threads > 0:
            cv2.setNumThreads(opencv_threads)
        self.bbox_smoother = BBoxSmoother(self.param("bbox_smoothing_seconds"))
        self.gallery_dir = Path(self.param("cases_dir")).expanduser().resolve()
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.image_changed = threading.Event()
        self.depth_frames = deque(maxlen=30)
        self.depth_infos = deque(maxlen=8)
        self.performance = {}
        self.model_timings = ModelTimings()
        self.sam3_event_stream = str(time.time_ns())
        self.sam3_event_sequence = 0
        self.sam3_events = deque(maxlen=101)
        self.frame_pool = SharedFramePool(slots=3)
        self.pending_render = None
        self.pending_annotated_image = None
        self.annotated_image_changed = threading.Event()
        self.render_changed = threading.Event()
        self.pending_publication = None
        self.publish_changed = threading.Event()
        self.stop = threading.Event()
        self.selected_class = ""
        self.generation = 0
        self.source = "camera"
        self.frame = None
        self.frame_seq = 0
        self.clear_object_debug()
        self.detections = []
        self.camera_frame = None
        self.camera_seq = 0
        self.camera_received_at = None
        self.camera_rate = ResultRate(window_seconds=3.0)
        self.camera_fps = None
        self.video_at = None
        self.video_error = None
        self.status = "Loading DINOv3 model…"
        self.backend_description = None
        self.reference_stride = None
        self.recognition_pub = self.create_publisher(
            Detection2DArray, "/medicine_recognition/detections", 10
        )
        self.model_timings_pub = self.create_publisher(
            String, "/medicine_recognition/model_timings",
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                       reliability=ReliabilityPolicy.RELIABLE),
        )
        self.model_timings_timer = self.create_timer(0.2, self.publish_model_timings)
        self.bbox3d_pub = self.create_publisher(
            Detection3DArray, "/medicine_recognition/detections_3d", 10
        )
        self.tag_pub = self.create_publisher(
            AprilTagDetectionArray, self.param("tag_topic"), 10
        )
        self.annotated_image_pub = self.create_publisher(
            ImageMsg, self.param("annotated_image_topic"),
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                       reliability=ReliabilityPolicy.RELIABLE),
        )
        try:
            self._start_workers()
        except BaseException:
            self.destroy_node()
            raise

    def _start_workers(self):
        self.image_stream = ImageStream(
            self.bridge, wake=self.render_changed.set,
            on_error=lambda error: self.get_logger().error(f"JPEG stream failed: {error}"),
        )
        # The browser bridge consumes the same ROS Image topic as RViz.
        self.annotated_image_subscription = self.create_subscription(
            ImageMsg, self.param("annotated_image_topic"), self.image_stream.update,
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                       reliability=ReliabilityPolicy.RELIABLE),
        )
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.create_subscription(ImageMsg, self.param("image_topic"), self.on_image, image_qos)
        if self.param("pose_enabled"):
            self.create_subscription(ImageMsg, self.param("depth_topic"), self.on_depth, image_qos)
            self.create_subscription(
                CameraInfo, self.param("depth_camera_info_topic"), self.on_depth_info, image_qos
            )
        self.app = create_app(self)
        self.server = make_server(self.param("host"), self.param("port"), self.app, threaded=True)
        self.server_worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_worker.start()
        self.image_publisher_worker = threading.Thread(target=self.run_image_publications, daemon=True)
        self.image_publisher_worker.start()
        self.publisher_worker = threading.Thread(target=self.run_publications, daemon=True)
        self.publisher_worker.start()
        self.renderer_worker = threading.Thread(target=self.run_rendering, daemon=True)
        self.renderer_worker.start()
        self.platform_pipeline.configure_geometry(self.param)
        self.bbox3d_worker = AsyncBBox3D(
            self.estimate_bboxes3d,
            max_age=self.param("bbox3d_max_age"),
            wake=self.image_changed.set,
            on_error=lambda error: self.get_logger().error(f"Background 3D boxes failed: {error}"),
        )
        self.worker = threading.Thread(target=self.run_grounded_worker, daemon=True)
        self.worker.start()
        self.get_logger().info(f"Browser: http://{self.param('host')}:{self.param('port')}")

    def set_readiness(self, state, message):
        with self.readiness_lock:
            payload = dict(state=state, message=message)
            if self.readiness == payload:
                return
            self.readiness = payload
        self.publish_readiness()

    def publish_readiness(self):
        with self.readiness_lock:
            payload = dict(self.readiness)
        worker = getattr(self, "worker", None)
        if worker is not None and not worker.is_alive() and payload["state"] != "error":
            payload = dict(state="error", message="Recognition worker stopped")
        self.readiness_pub.publish(String(data=json.dumps(payload)))

    def publish_model_timings(self):
        payload = self.model_timings.milliseconds()
        self.model_timings_pub.publish(String(data=json.dumps(payload, allow_nan=False)))

    def on_image(self, msg):
        with self.lock:
            self.camera_frame = msg
            self.camera_seq += 1
            self.camera_received_at = time.monotonic()
            self.camera_fps = self.camera_rate.update(self.camera_received_at, msg.header.frame_id)
            self.render_changed.set()
            if self.source != "camera":
                return
            self.frame = msg
            self.frame_seq += 1
            self.image_changed.set()

    def on_depth(self, msg):
        with self.lock:
            self.depth_frames.append(msg)

    def on_depth_info(self, msg):
        with self.lock:
            self.depth_infos.append(msg)

    def depth_inputs_for(self, frame):
        """Retain a matching depth snapshot without waiting in the RGB loop."""
        if not self.param("pose_enabled") or not isinstance(frame, ImageMsg):
            return None
        with self.lock:
            return matched_inputs(
                frame.header, frame.width, frame.height, self.depth_frames,
                self.depth_infos, self.param("pose_sync_tolerance")
            )

    def estimate_bboxes3d(self, detections, inputs):
        estimate_bboxes3d(self.bridge, self.param, self.platform_pipeline, detections, inputs)

    def worker_ticks(self):
        """Wake on input and periodically recheck non-image changes."""
        while not self.stop.is_set():
            self.image_changed.wait(0.1)  # Also recheck uploads, settings and gallery changes.
            self.image_changed.clear()
            if self.stop.is_set():
                return
            yield

    def capture_camera(self, prompt, box_threshold=None):
        with self.lock:
            detector = getattr(self, "grounding_detector", None)
            frame = self.camera_frame
            received_at = self.camera_received_at
            settings = dict(self.grounding_settings)
            region = getattr(self, "sam3_roi", DEFAULT_ROI)
        if detector is None:
            raise InferenceUnavailable("SAM3 is loading. Please try again shortly.")
        if frame is None or received_at is None or time.monotonic() - received_at >= 3:
            raise InferenceUnavailable("The camera is disconnected or its frame is stale. Check the camera feed.")
        detector.validate_prompt(prompt)
        image = Image.fromarray(self.bridge.imgmsg_to_cv2(frame, desired_encoding="rgb8")).copy()
        proposals = detector.detect(
            image, prompt, settings["box_threshold"] if box_threshold is None else box_threshold,
            self.param("grounding_max_detections"),
            min_width=self.param("grounding_min_bbox_width"),
            min_height=self.param("grounding_min_bbox_height"),
            min_area=self.param("grounding_min_bbox_area"),
            containment_threshold=self.param("grounding_containment_threshold"),
        )
        bounds = roi_bounds(*image.size, region) if region[:2] != (0, 0) else None
        return image, filter_roi(proposals, bounds)

    def queue_outputs(self, shared_frame, detections, header, gen, seq,
                      bbox3d_info, received_at, completed_at):
        """Caller holds self.lock and owns the input lease. Publication uses metadata.
        Retain one lease for rendering. Replacement releases the queued lease;
        the renderer releases a consumed lease, and shutdown releases any remainder.
        """
        self.pending_publication = (detections, header, gen)
        self.pending_publication_timing = dict(
            frame_id=seq, received_at=received_at, completed_at=completed_at)
        if self.pending_render is not None:
            self.pending_render[0].release()
        self.pending_render = (shared_frame.retain(), detections, header, gen,
                               seq, bbox3d_info, completed_at)
        self.render_changed.set()
        self.publish_changed.set()

    def run_publications(self):
        """DDS output cannot stall inference; pending output is a single latest slot."""
        while not self.stop.is_set():
            self.publish_changed.wait(0.1)
            self.publish_changed.clear()
            with self.lock:
                pending, self.pending_publication = self.pending_publication, None
                timing = getattr(self, "pending_publication_timing", None)
                self.pending_publication_timing = None
                generation = self.generation
            if pending is None:
                continue
            detections, header, gen = pending
            boxes = getattr(self, "bbox3d_worker", None)
            if boxes is not None:
                detections, _ = boxes.latest(detections)
            if gen != generation:
                continue
            try:
                publish_started = time.monotonic()
                topic_count = 0
                if header is None:
                    header = Header(stamp=self.get_clock().now().to_msg())
                if self.recognition_pub.get_subscription_count():
                    topic_count += 1
                    self.recognition_pub.publish(recognition_message(detections, header))
                if self.bbox3d_pub.get_subscription_count():
                    topic_count += 1
                    self.bbox3d_pub.publish(bboxes3d_message(detections, header))
                if self.tag_pub.get_subscription_count():
                    topic_count += 1
                    self.tag_pub.publish(apriltag_message(detections, header))
                published_at = time.monotonic()
                if timing is not None and topic_count:
                    received_at = timing["received_at"]
                    metrics = dict(
                        frame_id=timing["frame_id"], generation=gen,
                        published_at=published_at, topic_count=topic_count,
                        result_to_publish_ms=(published_at-timing["completed_at"])*1000,
                        dds_submit_ms=(published_at-publish_started)*1000,
                        callback_to_publish_ms=(None if received_at is None else
                                                (published_at-received_at)*1000),
                    )
                    with self.lock:
                        if gen == self.generation:
                            self.last_publication = metrics
                            if (self.performance.get("frame_id") == timing["frame_id"]
                                    and self.performance.get("completed_at") == timing["completed_at"]):
                                self.performance["ros_publication"] = metrics
            except Exception as error:
                self.get_logger().error(f"ROS result publication failed: {error}")

    def acquire_input_frame(self, pixels, *, frame_id, timestamp):
        """Reclaim a queued old overlay before letting rendering exhaust input slots."""
        frame = self.frame_pool.acquire(pixels, frame_id=frame_id, timestamp=timestamp)
        if frame is not None:
            return frame
        with self.lock:
            pending, self.pending_render = self.pending_render, None
        if pending is None:
            return None
        pending[0].release()
        return self.frame_pool.acquire(pixels, frame_id=frame_id, timestamp=timestamp)

    def run_rendering(self):
        """Draw once for the shared ROS Image output; keep the latest pending frame."""
        cached = None
        last_key, last_generation = None, None
        last_started = 0.0
        while not self.stop.is_set():
            self.render_changed.wait(0.2)
            self.render_changed.clear()
            delay = last_started + 1 / self.param("video_fps") - time.monotonic()
            if delay > 0 and self.stop.wait(delay):
                break
            with self.lock:
                job, self.pending_render = self.pending_render, None
                generation, source = self.generation, self.source
                camera, camera_seq = self.camera_frame, self.camera_seq
            try:
                # Exclude our own HTTP bridge subscription when no browser is watching.
                if (self.annotated_image_pub.get_subscription_count() <= 1
                        and not self.image_stream.has_readers):
                    cached, last_key = None, None
                    continue
                if generation != last_generation:
                    cached, last_key = None, None
                    last_generation = generation
                start = time.monotonic()
                render = render_display_overlay
                if job is not None:
                    frame, detections, header, gen, seq, info, completed_at = job
                    if gen == generation:
                        boxes = getattr(self, "bbox3d_worker", None)
                        if boxes is not None:
                            detections, measured_info = boxes.latest(detections)
                            info = measured_info or info
                        visible = any(item.get("bbox_visible", True) and
                                      (self.bbox_visualization_mode == "2d" or
                                       ("bbox3d" in item and info is not None))
                                      for item in detections)
                        if source == "upload" or (visible and time.monotonic() - completed_at < 3.0):
                            pixels = render(
                                frame.pixels, detections, info, self.bbox_visualization_mode)
                            cached = (pixels, header, ("result", gen, seq, completed_at),
                                      completed_at)
                        else:
                            cached = None
                if cached is not None and (source == "upload" or
                                           time.monotonic() - cached[3] < 3.0):
                    pixels, header, key, completed_at = cached
                elif source == "camera" and camera is not None:
                    key, completed_at = ("camera", generation, camera_seq), None
                    pixels, header = None, camera.header
                else:
                    continue
                # No codec refresh is needed; static frames stay in the HTTP JPEG cache.
                if key == last_key and not self.image_stream.needs_frame:
                    continue
                if pixels is None:
                    rgb = self.bridge.imgmsg_to_cv2(camera, desired_encoding="rgb8")
                    pixels = render(rgb, [], mode=self.bbox_visualization_mode)
                if header is None:
                    header = Header(stamp=self.get_clock().now().to_msg())
                finished = time.monotonic()
                with self.lock:
                    if generation != self.generation:
                        continue
                    self.pending_annotated_image = (pixels, header, generation)
                    self.annotated_image_changed.set()
                    self.video_at, self.video_error = finished, None
                    self.performance["video_render_ms"] = (finished - start) * 1000
                    if completed_at is not None and self.performance.get("completed_at") == completed_at:
                        self.performance["render_ms"] = (finished - start) * 1000
                        self.performance["result_to_overlay_ms"] = (finished - completed_at) * 1000
                last_started, last_key = start, key
            except Exception as error:
                with self.lock:
                    self.video_error = str(error)
                self.get_logger().error(f"Image rendering failed: {error}")
                self.stop.wait(0.5)
            finally:
                if job is not None:
                    job[0].release()

    def run_image_publications(self):
        """Publish only the newest raw overlay without blocking rendering or tracking."""
        while not self.stop.is_set():
            self.annotated_image_changed.wait(0.2)
            self.annotated_image_changed.clear()
            with self.lock:
                job, self.pending_annotated_image = self.pending_annotated_image, None
                generation = self.generation
            if job is None:
                continue
            pixels, header, gen = job
            if gen != generation or not self.annotated_image_pub.get_subscription_count():
                continue
            try:
                message = self.bridge.cv2_to_imgmsg(pixels, encoding="bgr8")
                message.header = header
                with self.lock:
                    if gen != self.generation:
                        continue
                # DDS may wait for a slow reader; never hold the shared state lock here.
                self.annotated_image_pub.publish(message)
            except Exception as error:
                self.get_logger().error(f"Raw image publication failed: {error}")

    def close_grounding_worker(self):
        detector = getattr(self, "grounding_detector", None)
        close = getattr(detector, "close", None)
        if close is not None:
            close()
        proposals = getattr(self, "proposal_tracker", None)
        if proposals is not None:
            proposals.close()
        boxes = getattr(self, "bbox3d_worker", None)
        if boxes is not None:
            boxes.close()

    def destroy_node(self):
        self.stop.set()
        self.image_changed.set()
        self.publish_changed.set()
        self.render_changed.set()
        self.annotated_image_changed.set()
        server_worker = getattr(self, "server_worker", None)
        if server_worker is not None and server_worker.is_alive():
            self.server.shutdown()
            server_worker.join()
        if hasattr(self, "server"):
            self.server.server_close()
        self.close_grounding_worker()
        if hasattr(self, "image_stream"):
            self.image_stream.close()
        for name, timeout in (("worker", 10), ("publisher_worker", 2),
                              ("image_publisher_worker", 3), ("renderer_worker", 3)):
            worker = getattr(self, name, None)
            if worker is not None and worker.is_alive():
                worker.join(timeout)
        with self.lock:
            if self.pending_render is not None:
                self.pending_render[0].release()
                self.pending_render = None
        self.frame_pool.close()
        platform = getattr(self, "platform_pipeline", None)
        if platform is not None:
            platform.close()
        return super().destroy_node()

    def clear_object_debug(self):
        self.object_debug_revision = getattr(self, "object_debug_revision", 0) + 1
        self.object_debug_snapshot = None
        self.object_debug_history = deque(maxlen=3)

    def run_grounded_worker(self):
        pipeline = getattr(self, "platform_pipeline", None)
        if pipeline is None:
            pipeline = self.platform_pipeline = load_pipeline(self.param("platform_config"))
        pipeline.run_live(self)


def main(args=None):
    rclpy.init(args=args)
    node = FewShotDetectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
