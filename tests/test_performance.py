from __future__ import annotations

from pathlib import Path

from test_privacy import create_video, NoBoxes, NoSpans
from video_privacy.config import Config
from video_privacy.detection import Detector
from video_privacy.video import process_video


def test_profile_preserves_all_frames_and_contains_no_content(tmp_path: Path) -> None:
    source=tmp_path/'source.mp4'
    create_video(source)
    config=Config((),(),())
    report=process_video(source,tmp_path/'out.mp4',tmp_path/'report.json',config,
                         Detector(config,NoSpans()),NoBoxes(),'drop')
    assert report['frames_processed']==3
    profile=report['performance']
    assert profile['width']==64 and profile['height']==64
    stages=profile['stage_seconds']
    for name in ('decode_seconds','pixel_conversion_seconds','guard_hash_seconds',
                 'ocr_roundtrip_seconds','classification_rules_seconds','mask_render_seconds',
                 'encode_mux_seconds','final_mux_seconds','output_verification_seconds'):
        assert stages[name]>=0
    assert max(stages.values())<=profile['processing_seconds']
    assert report['raw_text_in_report'] is False


def test_prefetch_overlaps_work_and_propagates_failure() -> None:
    from threading import Event
    import pytest
    from video_privacy.video import prefetch_one
    ready=Event()
    def source():
        yield 0
        ready.set()
        yield 1
        raise RuntimeError('fictional failure')
    ordered=prefetch_one(source())
    assert next(ordered)==0
    assert ready.wait(2), 'next frame must execute while current frame is handled'
    assert next(ordered)==1
    with pytest.raises(RuntimeError,match='fictional failure'):
        next(ordered)


def test_pipeline_preserves_masks_timestamps_and_single_frame_event(tmp_path: Path) -> None:
    from video_privacy.detection import Mask
    import av
    class SequenceDetector:
        notification_motion_guard_seconds=.75
        def __init__(self) -> None:
            self.calls=0
        def masks(self,boxes,width,height):
            self.calls+=1
            return [Mask((40,4,62,26),'auto_notification')] if self.calls==2 else []
    source=tmp_path/'source.mp4';create_video(source)
    observed=[]
    timestamps=[]
    for enabled in (False,True):
        regions=[]
        destination=tmp_path/f'out-{enabled}.mp4'
        process_video(source,destination,tmp_path/f'report-{enabled}.json',Config((),(),()),
                      SequenceDetector(),NoBoxes(),'drop',
                      region_observer=lambda i,m:regions.append((i,list(m))),prefetch_ocr=enabled)
        observed.append(regions)
        with av.open(str(destination)) as video:
            timestamps.append([frame.pts*frame.time_base for frame in video.decode(video=0)])
    assert observed[0]==observed[1]
    assert all(masks for _,masks in observed[1])
    assert timestamps[0]==timestamps[1]
