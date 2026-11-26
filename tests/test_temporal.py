from __future__ import annotations
from fractions import Fraction
import numpy as np
import pytest
from video_privacy.detection import Mask
from video_privacy.temporal import NotificationLookahead,ProtectedFrame,merge_corridors


def frame(t: float,masks: list[Mask] | None=None) -> ProtectedFrame:
    rgb=np.zeros((100,200,3),dtype=np.uint8)
    rgb[20,150]=1  # visible detail, including before a delayed recognition
    return ProtectedFrame(rgb,t,round(t*1000),Fraction(1,1000),[],masks or [])


def notification() -> Mask:
    return Mask((130,10,195,55),'auto_notification')


def test_future_recognition_protects_first_partial_entry_frame() -> None:
    guard=NotificationLookahead(.75);first=frame(0)
    assert guard.push(first)==[]
    assert guard.push(frame(.5,[notification()]))==[]
    released=guard.push(frame(.75))
    assert released[0] is first
    assert first.guards and first.guards[0].rect[0]<=190 and first.guards[0].rect[2]==200
    assert first.guards[0].rect[1]<=10 and first.guards[0].rect[3]>=55


def test_exit_remains_protected_after_text_is_no_longer_detectable() -> None:
    guard=NotificationLookahead(.75)
    guard.push(frame(0,[notification()]))
    last_visible=frame(.7);guard.push(last_visible)
    assert last_visible.guards
    absent=frame(.8);guard.push(absent)
    assert not absent.guards


def test_detection_at_exact_window_boundary_is_applied_before_release() -> None:
    guard=NotificationLookahead(.75);first=frame(0);guard.push(first)
    ready=guard.push(frame(.75,[notification()]))
    assert ready[0].guards


def test_later_event_does_not_claim_unbounded_lookback() -> None:
    guard=NotificationLookahead(.75);first=frame(0)
    guard.push(first);assert guard.push(frame(.8))==[first]
    guard.push(frame(1,[notification()]))
    assert not first.guards


@pytest.mark.parametrize('mask',[Mask((10,10,65,55),'auto_notification'),Mask((130,10,195,55),'model')])
def test_left_cards_and_non_notification_detections_do_not_expand(mask: Mask) -> None:
    guard=NotificationLookahead(.75);first=frame(0);guard.push(first)
    guard.push(frame(.5,[mask]));assert not first.guards


def test_finish_preserves_all_frames_order_and_timestamps() -> None:
    guard=NotificationLookahead(.75)
    frames=[frame(t) for t in [0,.1,.45,1.2,1.3]];output=[]
    for item in frames:output.extend(guard.push(item))
    output.extend(guard.finish())
    assert [id(x) for x in output]==[id(x) for x in frames]
    assert [x.pts for x in output]==[x.pts for x in frames]
    assert guard.bytes==0 and not guard.pending


def test_no_notification_never_adds_masks() -> None:
    guard=NotificationLookahead(.75);output=[]
    for t in [0,.5,1,2]:output.extend(guard.push(frame(t)))
    output+=guard.finish()
    assert not any(x.guards for x in output)


def test_memory_limit_stops_instead_of_releasing_unprotected_frames() -> None:
    guard=NotificationLookahead(.75,max_bytes=60000)
    guard.push(frame(0))
    with pytest.raises(RuntimeError,match='未略過畫面'):guard.push(frame(.1))
    assert len(guard.pending)==1


def test_separate_vertical_cards_do_not_merge_into_full_height_strip() -> None:
    masks=[Mask((130,10,200,30),'notification_motion_guard'),Mask((120,60,200,85),'notification_motion_guard')]
    assert len(merge_corridors(masks))==2
    merged=merge_corridors(masks+[Mask((125,20,200,35),'notification_motion_guard')])
    assert len(merged)==2


def test_disabled_guard_immediately_releases_manual_mode_frames() -> None:
    guard=NotificationLookahead(0);item=frame(0,[notification()])
    assert guard.push(item)==[item] and not item.guards


def test_render_pipeline_backfills_future_detection_and_keeps_vfr_order(tmp_path) -> None:
    import av
    from test_privacy import create_video,NoBoxes
    from video_privacy.config import Config
    from video_privacy.video import process_video
    class DelayedDetector:
        notification_motion_guard_seconds=.75
        def __init__(self):self.calls=0
        def masks(self,boxes,width,height):
            self.calls+=1
            return [Mask((40,4,62,26),'auto_notification')] if self.calls==2 else []
    source=tmp_path/'source.mp4';create_video(source,background_detail=True)
    observed=[]
    result=process_video(source,tmp_path/'result.mp4',tmp_path/'report.json',Config((),(),()),
                         DelayedDetector(),NoBoxes(),'drop',
                         region_observer=lambda index,masks:observed.append((index,masks)))
    assert result['frames_processed']==3
    assert [i for i,_ in observed]==[0,1,2]
    assert all(any(m.reason=='notification_motion_guard' for m in masks) for _,masks in observed)
    assert result['notification_temporal_protection']['guarded_frames']==3
    with av.open(str(source)) as before,av.open(str(tmp_path/'result.mp4')) as after:
        assert [f.pts*f.time_base for f in before.decode(video=0)]==[f.pts*f.time_base for f in after.decode(video=0)]


def test_exact_white_background_is_not_given_a_notification_guard() -> None:
    guard=NotificationLookahead(.75)
    before=frame(0);before.rgb[:]=255
    guard.push(before)
    guard.push(frame(.5,[notification()]))
    assert not before.guards
    after=frame(.6);after.rgb[:]=255
    guard.push(after)
    assert not after.guards


def test_one_changed_pixel_in_white_background_retains_notification_protection() -> None:
    guard=NotificationLookahead(.75)
    first=frame(0);first.rgb[:]=255;first.rgb[20,150]=254
    guard.push(first);guard.push(frame(.5,[notification()]))
    assert first.guards


@pytest.mark.parametrize('color',[(0,0,0),(180,180,180),(254,254,254),(17,18,23)])
def test_any_exact_uniform_background_needs_no_temporal_guard(color) -> None:
    guard=NotificationLookahead(.75)
    before=frame(0);before.rgb[:]=color
    guard.push(before);guard.push(frame(.5,[notification()]))
    assert not before.guards
