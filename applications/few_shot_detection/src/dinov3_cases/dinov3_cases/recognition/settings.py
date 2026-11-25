"""Application input validation and original-image proposal geometry."""
import math
import numpy as np
from dinov3_cases.tracking.kcf import validate_detection_interval

DEFAULT_PROMPT = "medicine bottle, medicine box."


def validate_settings(prompt, box_threshold, recognition_threshold, margin, detection_interval=0.0):
    """Validate external settings, including the accepted legacy margin field."""
    validate_detection_interval(detection_interval)
    if not isinstance(prompt, str) or not prompt.strip(" .\t\r\n") or len(prompt) > 1000:
        raise ValueError("Prompt must contain 1–1000 characters.")
    for name, value, upper in [
        ("box_threshold", box_threshold, 1),
        ("recognition_threshold", recognition_threshold, 1),
        ("recognition_margin", margin, 2),
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= upper
        ):
            raise ValueError(f"{name} must be a finite number between 0 and {upper}.")
    return prompt.strip().rstrip(".") + "."


def validate_bbox_limits(min_width=0, min_height=0, min_area=0):
    for name, value in (("min_width", min_width), ("min_height", min_height),
                        ("min_area", min_area)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")


def validate_containment_threshold(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("containment_threshold must be a finite number between 0 and 1.")


def clean_proposals(boxes, scores, labels, size, limit=20,
                    *, min_width=0, min_height=0, min_area=0, containment_threshold=0.95):
    """Clip XYXY boxes; remove contained boxes, then rank by score and limit."""
    validate_containment_threshold(containment_threshold)
    validate_bbox_limits(min_width, min_height, min_area)
    width, height = size
    candidates = []
    for box, score, label in zip(boxes, scores, labels):
        if len(box) != 4 or not np.isfinite([*box, score]).all():
            continue
        x0, y0, x1, y1 = box
        x0, y0 = max(0, math.floor(x0)), max(0, math.floor(y0))
        x1, y1 = min(width, math.ceil(x1)), min(height, math.ceil(y1))
        if x1 <= x0 or y1 <= y0:
            continue
        box_width, box_height = x1 - x0, y1 - y0
        area = box_width * box_height
        if box_width < min_width or box_height < min_height or area < min_area:
            continue
        candidates.append(
            dict(
                bbox=[x0, y0, x1, y1],
                detection_score=float(score),
                detection_label=str(label),
                area=area,
            )
        )
    # Prefer larger boxes when intersection / smaller area reaches the threshold, even
    # across prompt labels or when the smaller box has a higher score.
    # Equal-area duplicates keep the higher score. Filter before the result
    # limit so discarded small boxes cannot consume detection slots.
    outer_boxes = []
    for item in sorted(candidates, key=lambda x: (x["area"], x["detection_score"]),
                       reverse=True):
        a = item["bbox"]
        for other in outer_boxes:
            b = other["bbox"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            if containment_threshold > 0 and intersection / item["area"] >= containment_threshold:
                break
        else:
            outer_boxes.append(item)
    return sorted(outer_boxes, key=lambda item: item["detection_score"], reverse=True)[:limit]


