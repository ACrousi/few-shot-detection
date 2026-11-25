"""Real CPU KCF motion plus lifecycle/loss contracts; no model/GPU required."""
import subprocess
import sys
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from dinov3_cases.tracking.kcf import KCF
from dinov3_cases.tracking.async_tracker import _Tracks


def frame(x=50, second=False):
    rgb = np.full((240, 400, 3), 25, dtype=np.uint8)
    texture = np.random.default_rng(42).integers(30, 255, (48, 48, 3), dtype=np.uint8)
    rgb[80:128, x:x+48] = texture
    if second:
        rgb[160:208, 270:318] = 255-texture
    return rgb


def proposal(x=50, track_id=7):
    return dict(bbox=[x,80,x+48,128], label='sample', detection_score=.9, track_id=track_id)


def test_real_motion_ids_metadata_and_duplicate_frames():
    tracker = KCF(width=400, height=240)
    try:
        first = tracker.update(frame(), 0, [proposal()])[0]
        assert first['bbox_source']=='sam3'
        for i in range(1, 16):
            rows = tracker.update(frame(50+2*i), i)
            assert len(rows)==1
            row=rows[0]
            assert abs(row['bbox'][0]-(50+2*i)) <= 3
            assert row['track_id']==7 and row['tracker_id']==first['tracker_id']
            assert row['label']=='sample' and row['detection_score']==.9
            assert row['bbox_source']=='kcf'
            assert 'tracker_confidence' not in row
            assert tracker.update(frame(150), i) is rows
    finally: tracker.close()


def test_multiple_objects_metadata_removal_and_empty_reseed():
    tracker=KCF(width=400,height=240)
    rows=tracker.update(frame(second=True), 0,
                        [proposal(), dict(bbox=[270,160,318,208], label='other', track_id=8)])
    assert len(rows)==2
    del tracker.metadata[rows[0]['tracker_id']]
    assert [p['track_id'] for p in tracker.update(frame(second=True),1)]==[8]
    assert len(tracker.trackers)==1
    assert tracker.update(frame(),2,[])==[]
    assert tracker.update(frame(),3)==[]
    tracker.close();tracker.close()
    with pytest.raises(RuntimeError,match='closed'):tracker.update(frame(),4)


def test_resize_maps_back_to_source_coordinates():
    tracker=KCF(width=200,height=120)
    tracker.update(frame(),0,[proposal()])
    result=tracker.update(frame(54),1)[0]
    assert abs(result['bbox'][0]-54)<=4
    assert result['bbox'][2] <= 400
    # A resolution change without fresh detections invalidates old tracking state.
    assert tracker.update(np.zeros((300,500,3),np.uint8),2)==[]
    tracker.close()


def test_failed_update_hides_track_then_reacquires_before_grace():
    tracker=KCF()
    tracker.update(frame(),1,[proposal()])
    fake=Mock()
    fake.update.side_effect=[(False,(0,0,0,0)),(True,(52,80,48,48))]
    tracker.trackers[0]=fake
    tracks=_Tracks([tracker],now=1.,max_missed_seconds=2.)
    tracks.update(frame(),2,now=1.1)
    assert not tracks.proposals
    assert tracks.hidden[0]['track_id']==7
    tracks.update(frame(52),3,now=1.2)
    assert tracks.proposals[0]['track_id']==7
    assert not tracks.hidden
    tracks.close()


def test_failed_initialization_keeps_previous_context(monkeypatch):
    tracker=KCF();tracker.update(frame(),1,[proposal()])
    original=tracker.trackers.copy()
    fake=Mock();fake.init.side_effect=cv2.error('bad initialization')
    monkeypatch.setattr(tracker,'_create',lambda:fake)
    with pytest.raises(cv2.error):tracker.update(frame(),2,[proposal()])
    assert tracker.trackers==original and tracker.frame_key==1
    tracker.close()


@pytest.mark.parametrize('kwargs',[dict(width=0),dict(height=True),dict(detect_threshold=float('nan')),
                                   dict(detect_threshold=0),dict(detect_threshold=2)])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):KCF(**kwargs)


def test_invalid_input_and_boxes():
    tracker=KCF()
    with pytest.raises(ValueError):tracker.update(np.zeros((10,10),np.uint8),0)
    with pytest.raises(ValueError):tracker.update(frame(),0,[dict(bbox=[0,0,float('nan'),10])])
    assert tracker.update(frame(),0,[dict(bbox=[-20,-10,-1,5])])==[]
    tracker.close()


def test_tracking_import_and_execution_need_no_vendor_runtime():
    subprocess.run([sys.executable,'-c','''
import sys
from dinov3_cases.tracking.kcf import KCF
from dinov3_cases.tracking.async_tracker import AsyncProposals
tracker=KCF(); tracker.close()
assert not any(name in sys.modules for name in ('torch','tensorrt','rclpy'))
'''],check=True)
