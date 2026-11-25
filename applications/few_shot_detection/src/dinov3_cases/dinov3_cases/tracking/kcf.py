"""CPU OpenCV KCF contexts; one tracker per detection, CPU only."""
import math
import cv2
import numpy as np


def check_kcf():
    if not callable(getattr(cv2, "TrackerKCF_create", None)):
        raise RuntimeError("OpenCV KCF is unavailable. Install opencv-contrib-python (not opencv-python).")


def validate_detection_interval(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("detection_interval must be finite seconds >= 0")


def validate_tracker_settings(width, height, detect_threshold):
    for name, value in (("tracker_width", width), ("tracker_height", height)):
        if type(value) is not int or not 32 <= value <= 4096:
            raise ValueError(f"{name} must be an integer in [32, 4096]")
    if (isinstance(detect_threshold, bool) or not isinstance(detect_threshold, (int, float))
            or not math.isfinite(detect_threshold) or not 0 < detect_threshold <= 1):
        raise ValueError("kcf_detect_threshold must be finite in (0, 1]")


class KCF:
    """Single-owner context implementing update/proposals/metadata/frame_key/close.

    Input: uint8 RGB HWC; output: original-image XYXY. Tracker IDs are context-local.
    Metadata is scheduler-owned; removing an ID also removes its KCF instance.
    Failed tracks stay hidden until detection reacquires them or the scheduler
    expires their ownership. OpenCV exposes success, not a calibrated confidence.
    """
    def __init__(self, width=640, height=384, detect_threshold=0.5):
        check_kcf()
        validate_tracker_settings(width, height, detect_threshold)
        self.width, self.height, self.detect_threshold = width, height, detect_threshold
        self.metadata, self.trackers = {}, {}
        self.proposals = []
        self.frame_key = None
        self.source_size = None
        self.initialized = False
        self.closed = False

    def _create(self):
        params = cv2.TrackerKCF_Params()
        params.detect_thresh = self.detect_threshold
        return cv2.TrackerKCF_create(params)

    def update(self, pixels, frame_key, detections=None):
        if self.closed:
            raise RuntimeError("KCF context is closed")
        if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
                or pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 2):
            raise ValueError("KCF expects nonempty uint8 HWC RGB")
        if self.initialized and frame_key == self.frame_key and detections is None:
            return self.proposals
        height, width = pixels.shape[:2]
        if self.source_size is not None and self.source_size != (width, height) and detections is None:
            # Old coordinates cannot be safely reused across a resolution change.
            self.trackers.clear()
            self.metadata.clear()
            self.proposals = []
            self.frame_key = frame_key
            self.source_size = (width, height)
            return self.proposals
        # Keep aspect ratio and cap CPU work; do not stretch source geometry.
        scale = min(self.width / width, self.height / height, 1.0)
        tw, th = max(2, round(width * scale)), max(2, round(height * scale))
        sx, sy = tw / width, th / height
        rgb = cv2.resize(pixels, (tw, th), interpolation=cv2.INTER_AREA) if (tw, th) != (width, height) else pixels
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        seeded = detections is not None
        if seeded:
            if len(detections) > 150:
                raise ValueError("KCF supports at most 150 detections per context")
            trackers, metadata = {}, {}
            try:
                for index, proposal in enumerate(detections):
                    box = np.asarray(proposal["bbox"], dtype=float)
                    if box.shape != (4,) or not np.isfinite(box).all():
                        raise ValueError("KCF detection bbox must be finite XYXY")
                    x0, y0 = max(0, math.floor(box[0]*sx)), max(0, math.floor(box[1]*sy))
                    x1, y1 = min(tw, math.ceil(box[2]*sx)), min(th, math.ceil(box[3]*sy))
                    if x1-x0 < 2 or y1-y0 < 2:
                        continue
                    tracker = self._create()
                    if tracker.init(bgr, (x0, y0, x1-x0, y1-y0)) is False:
                        raise RuntimeError("KCF initialization failed")
                    # OpenCV KCF learns its first template on the first update.
                    ok, _ = tracker.update(bgr)
                    if not ok:
                        raise RuntimeError("KCF template initialization failed")
                    trackers[index] = tracker
                    metadata[index] = dict(proposal, bbox=[max(0,math.floor(box[0])),max(0,math.floor(box[1])),
                                                         min(width,math.ceil(box[2])),min(height,math.ceil(box[3]))])
            except BaseException:
                trackers.clear()
                raise
            self.trackers, self.metadata = trackers, metadata
        else:
            self.trackers = {key: tracker for key, tracker in self.trackers.items() if key in self.metadata}
        output = []
        for key, tracker in self.trackers.items():
            item = self.metadata[key]
            if seeded:
                box = item["bbox"]
            else:
                ok, (x, y, w, h) = tracker.update(bgr)
                if not ok or not np.isfinite([x,y,w,h]).all():
                    continue
                box = [max(0,round(x/sx)), max(0,round(y/sy)),
                       min(width,round((x+w)/sx)), min(height,round((y+h)/sy))]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
            result = dict(item, bbox=list(box), tracker_id=key,
                          area=(box[2]-box[0])*(box[3]-box[1]),
                          track_status="confirmed" if seeded else "coasting",
                          bbox_source="sam3" if seeded else "kcf")
            output.append(result)
            # Retain the last valid box for hidden-track association on reacquisition.
            self.metadata[key] = dict(result)
        self.proposals, self.frame_key = output, frame_key
        self.source_size, self.initialized = (width, height), True
        return output

    def close(self):
        self.trackers.clear()
        self.metadata.clear()
        self.proposals = []
        self.closed = True
