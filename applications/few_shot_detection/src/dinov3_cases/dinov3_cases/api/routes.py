"""HTTP routes for live recognition controls, results, and gallery management."""

import time
from io import BytesIO
from flask import Flask, jsonify, request, Response, render_template
from PIL import Image
from dinov3_cases.runtime import MODEL_ID, GROUNDING_MODEL_ID, INPUT_SIZE
from dinov3_cases.api.image_manager import register_image_manager
from dinov3_cases.inference.sam3_roi import DEFAULT_ROI, validate_roi, roi_bounds
from dinov3_cases import live_state
from dinov3_cases.recognition.gallery_files import group_path as case_path, image_path, load_image, gallery_groups, gallery_images, save_image


def create_app(node):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

    @app.after_request
    def wake_after_update(response):
        if request.method != "GET":
            node.image_changed.set()
            node.render_changed.set()
        return response

    def images_changed(group):
        with node.lock:
            live_state.gallery_changed(node)

    @app.errorhandler(live_state.InferenceUnavailable)
    def unavailable(error):
        return jsonify(error=str(error)), 503

    def capture_threshold():
        with node.lock:
            return node.grounding_settings["box_threshold"]

    register_image_manager(app, node.gallery_dir, images_changed, capture=node.capture_camera,
                           capture_threshold=capture_threshold)

    @app.errorhandler(ValueError)
    def bad_request(error):
        return jsonify(error=str(error)), 400

    @app.get("/")
    def index():
        return render_template("cases.html", input_size=INPUT_SIZE,
                               bbox_mode=node.bbox_visualization_mode)

    @app.get("/api/status")
    def status():
        cases = [path.name for path in gallery_groups(node.gallery_dir)]
        with node.lock:
            age = (
                None
                if node.camera_received_at is None
                else time.monotonic() - node.camera_received_at
            )
            platform = getattr(node, "platform_pipeline", None)
            platform_status = platform.status() if platform is not None else dict(
                platform="nvidia", grounding_backend="trt", inference_backend="trt",
                grounding_execution="subprocess", grounding_engine_path=None)
            return jsonify(
                **platform_status,
                cases=cases,
                selected=node.selected_class,
                source=node.source,
                status=node.status,
                performance=node.performance,
                model_timings=(node.model_timings.milliseconds()
                               if getattr(node, "model_timings", None) is not None else {}),
                sam3_inference_events=dict(
                    stream=getattr(node, "sam3_event_stream", None),
                    events=list(getattr(node, "sam3_events", ())),
                ),
                overlay_ready=node.video_at is not None,
                bbox_visualization_mode=node.bbox_visualization_mode,
                grounding=node.grounding_settings,
                grounding_model_id=GROUNDING_MODEL_ID,
                grounding_schedule=(node.proposal_tracker.scheduler_metrics()
                                    if getattr(node, "proposal_tracker", None) is not None else None),
                grounding_precision=getattr(getattr(node, "grounding_detector", None), "precision", None),
                grounding_worker=getattr(
                    getattr(node, "grounding_detector", None), "worker_info", None
                ),
                object_max_side=INPUT_SIZE,
                object_square_input=True,
                reference_revision=node.generation,
                reference_stride=node.reference_stride,
                model_id=MODEL_ID,
                backend_description=node.backend_description,
                camera_topic=node.param("image_topic"),
                camera_age=age,
                camera_fps=getattr(node, "camera_fps", None),
                sam3_roi=region_state(),
                camera_connected=age is not None and age < 3,
                video_error=node.video_error,
                overlay_age=(
                    None if node.video_at is None else time.monotonic() - node.video_at
                ),
            )

    @app.get("/api/references")
    def references():
        name = request.args.get("case", "")
        folder = case_path(node.gallery_dir, name)
        if not folder.is_dir():
            raise ValueError("Case does not exist.")
        with node.lock:
            stride = node.reference_stride
        if stride is None:
            return jsonify(error="The model is loading. Patch grids will be available shortly."), 503
        items = []
        for path in gallery_images(folder):
            item = dict(name=path.name, version=f"{path.stat().st_mtime_ns}-{path.stat().st_size}")
            try:
                original = load_image(path, mode="RGB")
                reference = node.platform_pipeline.reference_preview(original)
                item.update(
                    original_size=list(original.size),
                    size=list(reference.size),
                    rows=reference.height // stride,
                    columns=reference.width // stride,
                )
            except (ValueError, OSError, Image.DecompressionBombError) as error:
                item["error"] = str(error)
            items.append(item)
        return jsonify(case=name, patch_size=stride, images=items)

    @app.get("/api/references/grid/<group>/<filename>")
    def reference_grid(group, filename):
        path = image_path(node.gallery_dir, group, filename)
        with node.lock:
            stride = node.reference_stride
        if stride is None:
            return jsonify(error="The model is loading."), 503
        reference = node.platform_pipeline.reference_preview(load_image(path, mode="RGB"), stride)
        data = BytesIO()
        reference.save(data, format="PNG")
        return Response(
            data.getvalue(), mimetype="image/png", headers={"Cache-Control": "no-store"}
        )

    @app.post("/api/grounding")
    def grounding_settings():
        payload = request.get_json(silent=True)
        if (
            not isinstance(payload, dict)
            or not payload
            or set(payload) - set(node.grounding_settings)
        ):
            raise ValueError("Expected prompt, detection_interval and/or grounding/recognition thresholds.")
        with node.lock:
            settings = live_state.set_grounding_settings(node, payload)
        return jsonify(ok=True, grounding=settings)

    def region_state():
        region = getattr(node, "sam3_roi", DEFAULT_ROI)
        frame = getattr(node, "camera_frame", None)
        native_size = ([frame.width, frame.height]
                       if frame is not None and hasattr(frame, "width") else None)
        bounds = list(roi_bounds(*native_size, region)) if native_size else None
        return dict(zip(("width", "height", "center_x", "center_y"), region),
                    native_size=native_size, bounds=bounds,
                    enabled=region[:2] != (0, 0))

    @app.get("/api/sam3-roi")
    def get_region():
        with node.lock:
            return jsonify(region_state())

    @app.post("/api/sam3-roi")
    def set_region():
        payload = request.get_json(silent=True)
        keys = ("width", "height", "center_x", "center_y")
        if not isinstance(payload, dict) or not payload or set(payload) - set(keys):
            raise ValueError("Expected region width, height, center_x and/or center_y.")
        with node.lock:
            previous = getattr(node, "sam3_roi", DEFAULT_ROI)
            region = validate_roi(*(payload.get(key, old) for key, old in zip(keys, previous)))
            live_state.set_region(node, region, previous)
            state = region_state()
        return jsonify(state)

    @app.post("/api/cases")
    def upload():
        selected_class = request.form.get("case", "")
        folder = case_path(node.gallery_dir, selected_class)
        files = request.files.getlist("images")
        if not files:
            raise ValueError("Choose at least one picture.")
        # Validate all pictures before modifying the case.
        images = [load_image(f.stream, mode="RGB") for f in files]
        folder.mkdir(exist_ok=True)
        for i, image in enumerate(images):
            dest = folder / f"{time.time_ns()}_{i}.png"
            save_image(image, dest)
        with node.lock:
            live_state.select_class(node, selected_class, rebuild=True)
        return jsonify(ok=True, count=len(images))

    @app.post("/api/select")
    def select():
        selected_class = (request.get_json(silent=True) or {}).get("case", "")
        folder = case_path(node.gallery_dir, selected_class)
        if not folder.is_dir():
            raise ValueError("Case does not exist.")
        with node.lock:
            live_state.select_class(node, selected_class)
        return jsonify(ok=True)

    @app.post("/api/source")
    def source():
        mode = request.form.get("source", "camera")
        if mode not in ("camera", "upload"):
            raise ValueError("Invalid source.")
        image = None
        if mode == "upload":
            if "image" not in request.files:
                raise ValueError("Choose a target picture.")
            image = load_image(request.files["image"].stream, mode="RGB")
        with node.lock:
            live_state.set_source(node, mode, image)
        return jsonify(ok=True)

    @app.get("/image_annotated.mjpg")
    def annotated_image():
        return Response(
            node.image_stream.multipart(),
            content_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/detections")
    def detections():
        with node.lock:
            return jsonify(detections=node.detections)

    @app.get("/api/object_visualizations")
    def object_visualizations():
        reference_case = request.args.get("reference_case", "")
        with node.lock:
            revision, snapshot = node.object_debug_revision, node.object_debug_snapshot
            if snapshot is not None and not any(
                v == revision for v, _ in node.object_debug_history
            ):
                node.object_debug_history.append((revision, snapshot))
        if snapshot is None:
            return jsonify(
                revision=revision,
                objects=[],
                reference_cases=[],
                reference_case="",
                ready=False,
            )
        try:
            items, _ = snapshot.render(reference_case)
        except ValueError as error:
            return jsonify(error=str(error), reference_cases=snapshot.reference_cases), 400
        return jsonify(
            revision=revision,
            objects=items,
            reference_cases=snapshot.reference_cases,
            reference_case=reference_case,
            ready=True,
        )

    @app.get("/debug/objects/<int:revision>/<int:index>/<view>.jpg")
    def object_debug_view(revision, index, view):
        with node.lock:
            snapshot = next(
                (
                    snapshot
                    for version, snapshot in node.object_debug_history
                    if version == revision
                ),
                None,
            )
            if snapshot is None:
                return Response(status=410)
        _, images = snapshot.render(request.args.get("reference_case", ""))
        data = images.get((index, view))
        return Response(
            data or b"",
            status=200 if data else 404,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    return app
