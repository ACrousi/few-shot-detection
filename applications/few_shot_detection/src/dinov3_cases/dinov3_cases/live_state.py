"""Recognition state transitions for live input and gallery controls.

Callers must hold node.lock. Keep each setting/source update and invalidation in
one critical section so workers cannot publish results from the old generation.
Validate changes before committing them. Decode uploaded images before taking the lock.
"""


from dinov3_cases.recognition.settings import validate_settings
from dinov3_cases.tracking.kcf import check_kcf


class InferenceUnavailable(RuntimeError):
    """The requested operation needs a ready camera frame or detector."""


def invalidate_results(node):
    """Discard output from the previous input revision, including MJPEG/debug data."""
    node.generation += 1
    node.video_at = None
    node.image_stream.reset()
    node.clear_object_debug()
    node.detections = []


def set_grounding_settings(node, changes):
    settings = {**node.grounding_settings, **changes}
    settings["prompt"] = validate_settings(**settings)
    check_kcf()
    detector = getattr(node, "grounding_detector", None)
    if detector is None and "prompt" in changes:
        raise InferenceUnavailable("SAM3 is loading. Please try again once it is ready.")
    if detector is not None:
        detector.validate_prompt(settings["prompt"])
    node.grounding_settings = settings
    invalidate_results(node)
    node.status = "Settings applied: Preparing first detection and recognition…"
    return settings


def set_region(node, region, previous):
    """Reapplying the same region preserves valid results."""
    if region != previous:
        node.sam3_roi = region
        invalidate_results(node)
        node.status = "SAM3 region applied: Preparing first detection and recognition…"


def gallery_changed(node):
    invalidate_results(node)
    node.status = "Building reference features…"


def select_class(node, selected_class, *, rebuild=False):
    node.selected_class = selected_class
    if rebuild:
        gallery_changed(node)


def set_source(node, mode, image):
    node.source = mode
    node.frame = image
    node.frame_seq += 1
    invalidate_results(node)
