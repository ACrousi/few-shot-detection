"""NVIDIA live pipeline: scheduling, detection, recognition, tracking and result assembly."""
import time
import numpy as np
from PIL import Image
from sensor_msgs.msg import Image as ImageMsg
from dinov3_cases.runtime import INPUT_SIZE
from dinov3_cases.timing import ResultRate
from dinov3_cases.tracking.async_tracker import AsyncProposals
from dinov3_cases.inference.sam3_roi import DEFAULT_ROI, roi_bounds, filter_roi
from .patches import feature_stride
from .gallery_matching import recognize_objects
from .diagnostics import ObjectVisualizationSnapshot

def run_live_inference(node, pipeline):
    try:
        pipeline.load_models()
        encoder = pipeline.encoder
        stride = feature_stride(encoder.model)
        with node.lock:
            node.reference_stride = stride
            node.backend_description = encoder.description
            node.status = "Loading official SAM3 model…"
        node.set_readiness("loading", "Loading SAM3…")
        detector = pipeline.detector
        node.grounding_detector = detector
        if node.stop.is_set():
            node.close_grounding_worker()
            return
        with node.lock:
            detector.validate_prompt(node.grounding_settings["prompt"])
            node.status = "Models ready: Preparing first detection and recognition…"
    except Exception as error:
        node.close_grounding_worker()
        pipeline.close()
        with node.lock:
            node.status = f"Model load failed: {type(error).__name__}: {error}"
        node.set_readiness("error", node.status)
        node.get_logger().error(node.status)
        return
    gallery = pipeline.gallery
    processed = None
    completed_frame = None
    last_rgb_completed_at = None
    last_rgb_context = None
    next_upload_refresh = 0.0
    proposal_tracker = AsyncProposals(
        wake=node.image_changed.set,
        detect_threshold=node.param("kcf_detect_threshold"),
        width=node.param("tracker_width"),
        height=node.param("tracker_height"),
        containment_threshold=node.param("grounding_containment_threshold"),
        max_missed_seconds=node.param("tracker_max_missed_seconds"),
        defer_recognition=True,
        model_timings=node.model_timings,
    )
    node.proposal_tracker = proposal_tracker
    node.set_readiness("warming_up", "Models loaded; waiting for an image to initialize tracking and recognition…")
    last_recognition = None
    had_detections = False
    result_rate = ResultRate(window_seconds=3.0)
    max_side = INPUT_SIZE
    for _ in node.worker_ticks():
        shared_frame = None
        with node.lock:
            gen, frame, seq = node.generation, node.frame, node.frame_seq
            source = node.source
            settings = dict(node.grounding_settings)
            received_at = node.camera_received_at if source == "camera" else None
            region = getattr(node, "sam3_roi", DEFAULT_ROI)
        try:
            # Static uploads have no new frames to trigger periodic model jobs.
            if source == "upload" and time.monotonic() >= next_upload_refresh:
                with proposal_tracker.lock:
                    proposal_tracker.force_detection = True
                next_upload_refresh = time.monotonic() + 1.0
            if frame is None:
                with node.lock:
                    if gen == node.generation:
                        node.status = "Ready: Waiting for target image…"
                continue
            if processed == (gen, seq):
                # Fold background completions into the next live RGB. Do not
                # recopy, retrack, publish or encode an already processed frame.
                # Flush after 200 ms without input so paused cameras still get
                # their first 3D boxes and completed labels; uploads flush now.
                if (source == "camera" and received_at is not None
                        and time.monotonic() - received_at < 0.2):
                    continue
                if (not proposal_tracker.has_result()
                        and not (node.param("pose_enabled") and isinstance(frame, ImageMsg)
                                 and node.bbox3d_worker.has_result())):
                    continue
            start = time.monotonic()
            header = frame.header if isinstance(frame, ImageMsg) else None
            image_size = (frame.width, frame.height) if header is not None else frame.size
            bounds = (roi_bounds(*image_size, region)
                      if header is not None and region[:2] != (0, 0) else None)
            context = (gen, image_size, header.frame_id if header is not None else None)
            timing_revision = node.model_timings.set_context((context, tuple(settings.items())))
            if isinstance(frame, ImageMsg):
                pixels = node.bridge.imgmsg_to_cv2(frame, desired_encoding="rgb8")
            else:
                pixels = np.asarray(frame.convert("RGB"))
            shared_frame = node.acquire_input_frame(pixels, frame_id=seq, timestamp=received_at)
            if shared_frame is None:
                continue
            image_ready_at = time.monotonic()
            # Preserve synchronized depth before first-result inference waits.
            depth_inputs = node.depth_inputs_for(frame)
            depth_ready_at = time.monotonic()
            depth_wait_ms = (depth_ready_at - image_ready_at) * 1000
            # Capture settings by value: the job may outlive this loop iteration.
            request = dict(
                prompt=settings["prompt"], box_threshold=settings["box_threshold"],
                limit=node.param("grounding_max_detections"),
                min_width=node.param("grounding_min_bbox_width"),
                min_height=node.param("grounding_min_bbox_height"),
                min_area=node.param("grounding_min_bbox_area"),
                # Containment runs once on the final merged KCF tracks.
                containment_threshold=0.0,
            )

            def detect(frame, request=request, frame_key=(gen, seq), revision=timing_revision,
                       bounds=bounds):
                started_at = time.monotonic()
                with node.model_timings.measure("sam3", frame_key, revision) as timing:
                    proposals = detector.detect(frame, **request)
                    timing["duration_ms"] = detector.last_metrics["worker_detect_ms"]
                with node.lock:
                    node.sam3_event_sequence += 1
                    node.sam3_events.append(dict(
                        sequence=node.sam3_event_sequence, started_at=started_at,
                    ))
                return filter_roi(proposals, bounds), dict(getattr(detector, "last_metrics", {}))

            def recognize_impl(seed_pixels, proposals, seed_key, settings=settings):
                # Geometry is published before this background callback finishes.
                # All gallery GPU work and diagnostics stay off the KCF loop.
                started = time.monotonic()
                gallery.refresh(node.gallery_dir, encoder, max_side, stride)
                diagnostics = []
                detections = recognize_objects(
                    Image.fromarray(seed_pixels), proposals, encoder, gallery, max_side, stride,
                    settings["recognition_threshold"], diagnostics=diagnostics,
                    defer_diagnostics=True,
                )
                snapshot = ObjectVisualizationSnapshot(
                    [dict(item, bbox=list(item["bbox"])) for item in detections],
                    diagnostics, gallery, max_side, stride,
                )
                return detections, dict(
                    frame_id=seed_key[1] if proposals else None,
                    snapshot=snapshot,
                    class_count=len(set(gallery.labels)),
                    reference_count=len(gallery.labels),
                    recognize_ms=(time.monotonic() - started) * 1000,
                )

            def recognize(seed_pixels, proposals, seed_key, revision=timing_revision,
                          implementation=recognize_impl):
                with node.model_timings.measure("dinov3", seed_key, revision):
                    return implementation(seed_pixels, proposals, seed_key)

            with node.readiness_lock:
                starting = node.readiness["state"] == "warming_up"
            if starting:
                node.set_readiness("warming_up", "Preparing first SAM3 / KCF / DINOv3 result…")
            result = proposal_tracker.update(
                shared_frame.pixels, (gen, seq), context,
                settings["detection_interval"], detect, frame=shared_frame, recognize=recognize,
                wait_for_initial_result=True,
            )
            if result is None:
                processed = (gen, seq)
                continue  # Keep admitting new RGB frames without publishing empty results.
            proposals, proposal_source = result
            detected_at = time.monotonic()
            proposals = node.bbox_smoother.update(proposals, (gen, seq), context)
            # A track may leave the region between SAM3 refreshes.
            proposals = filter_roi(proposals, bounds)
            # Classification fields follow individual proposals through KCF,
            # including partial loss. Never zip survivors with an old label list.
            detections = proposals
            recognition = proposal_tracker.last_recognition
            recognition_info = recognition or dict(
                frame_id=None, snapshot=None, class_count=len(set(gallery.labels)),
                reference_count=len(gallery.labels), recognize_ms=0.0)
            recognition_changed = (recognition is not last_recognition
                                   or (had_detections and not detections))
            had_detections = bool(detections)
            last_recognition = recognition
            recognition_frame_id = recognition_info["frame_id"] if detections else None
            recognition_source = ("none" if not detections else
                                  "dino" if recognition_changed else "cached")
            if any(item.get("recognition_pending") for item in detections):
                recognition_source = "pending"
            recognized_at = time.monotonic()
            with node.lock:
                if gen != node.generation:
                    continue
                for item in detections:
                    item["bbox_visible"] = True
            bbox3d_schedule = None
            bbox3d_info = depth_inputs[1] if depth_inputs else None
            if node.param("pose_enabled") and header is not None:
                if depth_inputs is None:
                    depth_inputs = node.depth_inputs_for(frame)
                detections, measured_info, bbox3d_schedule = node.bbox3d_worker.update(
                    detections, depth_inputs, header=header,
                    frame_key=(gen, seq), context=context, captured_at=received_at,
                )
                bbox3d_info = measured_info or bbox3d_info
            else:
                for item in detections:
                    item["bbox3d_status"] = "disabled" if not node.param("pose_enabled") else "no_camera_depth"
            bbox3d_at = time.monotonic()
            # Diagnostic crops/features stay on their recognition frame;
            # never relabel an old crop as the current KCF image.
            object_snapshot = (recognition_info["snapshot"]
                               if recognition_changed and detections else None)
            with node.lock:
                if gen != node.generation:
                    continue
                if recognition_changed:
                    node.object_debug_revision += 1
                    node.object_debug_snapshot = object_snapshot
                node.detections = detections
                completed_at = time.monotonic()
                new_rgb_result = completed_frame != (gen, seq)
                result_fps = node.performance.get("result_fps")
                result_fps_instant = None
                if new_rgb_result:
                    if last_rgb_context == context and last_rgb_completed_at is not None:
                        result_fps_instant = 1 / max(1e-9, completed_at - last_rgb_completed_at)
                    result_fps = result_rate.update(completed_at, context)
                    completed_frame = (gen, seq)
                    last_rgb_completed_at, last_rgb_context = completed_at, context
                node.publication_drops = getattr(node, "publication_drops", 0) + int(
                    node.pending_publication is not None)
                node.performance = dict(
                    completed_at=completed_at,
                    result_fps=result_fps,
                    result_fps_window_seconds=result_rate.window_seconds,
                    result_fps_instant=result_fps_instant,
                    new_rgb_result=new_rgb_result,
                    grounding_ipc=(dict(proposal_tracker.last_metrics)
                                   if proposal_source == "sam3" else {}),
                    grounding_schedule=proposal_tracker.scheduler_metrics(),
                    ros_publication=getattr(node, "last_publication", None),
                    bbox_source=proposal_source,
                    recognition_source=recognition_source,
                    recognition_frame_id=recognition_frame_id,
                    detection_interval=settings["detection_interval"],
                    image_prepare_ms=(image_ready_at - start) * 1000,
                    shared_copy_ms=shared_frame.copy_ms,
                    depth_wait_ms=depth_wait_ms,
                    proposal_update_ms=(detected_at - depth_ready_at) * 1000,
                    detect_ms=(detected_at - start) * 1000,
                    recognize_ms=(recognition_info["recognize_ms"] if recognition_changed else 0.0),
                    bbox3d_ms=(bbox3d_at - recognized_at) * 1000,
                    bbox3d_count=sum("bbox3d" in item for item in detections),
                    bbox3d_schedule=bbox3d_schedule,
                    render_ms=None,
                    frame_id=seq,
                    input_skipped_frames=(max(0, seq-processed[1]-1)
                                          if processed and processed[0] == gen else 0),
                    publication_drops=node.publication_drops,
                    processing_ms=(completed_at - start) * 1000,
                    callback_to_result_ms=(
                        None if received_at is None else (completed_at - received_at) * 1000
                    ),
                )
                node.status = (
                    f"{('SAM3 Torch-TensorRT' if proposal_source == 'sam3' else 'KCF')} → "
                    f"{encoder.description}{'' if recognition_changed else ' (cached)'} | {len(detections)} objects | "
                    f"{recognition_info['class_count']} classes / {recognition_info['reference_count']} references | "
                    f"{time.monotonic()-start:.2f}s inference"
                )
                node.queue_outputs(shared_frame, detections, header, gen, seq,
                                   bbox3d_info, received_at, completed_at)
            if proposal_tracker.recognition_error:
                node.set_readiness("error", "Recognition failed: " + proposal_tracker.recognition_error)
            elif recognition is not None:
                node.set_readiness("ready", "Recognition ready")
            processed = (gen, seq)
        except Exception as error:
            with node.lock:
                if gen == node.generation:
                    node.clear_object_debug()
                    node.detections = []
                    node.video_at = None
                    node.image_stream.reset()
                    node.status = f"Error: {type(error).__name__}: {error}"
            node.set_readiness("error", f"Recognition failed: {error}")
            node.get_logger().error(str(error))
            # Retry failed files/model calls at a bounded rate, including static uploads.
            node.stop.wait(1.0)
        finally:
            if shared_frame is not None:
                shared_frame.release()
