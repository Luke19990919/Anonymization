from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import subprocess

import av
import imageio_ffmpeg
import numpy as np
import pytest

from video_privacy.config import Config, Participant, Region, letter_id, parse_config
from video_privacy.detection import Box, Detector, Mask
from video_privacy.video import paint, process_video


class NoSpans:
    def spans(self, text: str) -> list[tuple[int, int, str]]:
        return []


class NoBoxes:
    def read(self, rgb: np.ndarray) -> list[Box]:
        return []


def create_video(path: Path, timestamps: tuple[int, ...] = (0, 100, 350), *, background_detail: bool=False) -> None:
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('libx264', rate=4)
        stream.width, stream.height, stream.pix_fmt = 64, 64, 'yuv420p'
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for index, pts in enumerate(timestamps):
            rgb = np.full((64,64,3), 180, dtype=np.uint8)
            if background_detail:rgb[10,50]=0
            if index == 1:
                rgb[4:26,4:26] = [255,20,20]
            frame = av.VideoFrame.from_ndarray(rgb,format='rgb24')
            frame.pts, frame.time_base = pts, Fraction(1,1000)
            for packet in stream.encode(frame): output.mux(packet)
        for packet in stream.encode(): output.mux(packet)


def config() -> Config:
    return Config((), (), (Region('notification','notification',(0,0,0.5,0.5)),))


def test_unicode_names_and_consistent_distinct_aliases() -> None:
    cfg=Config((Participant(('王小明','Wang Example'),'受測者 A'),Participant(('陳小華',),'受測者 B')),(),())
    detector=Detector(cfg,NoSpans())
    for text, replacement in [('王 小 明','受測者 A'),('陳小華','受測者 B'),('王小明','受測者 A')]:
        masks=detector.masks([Box((0,0,80,30),text,.95)],200,200)
        assert replacement in [m.replacement for m in masks]


def test_name_split_across_ocr_boxes_keeps_alias() -> None:
    cfg=Config((Participant(('王小明',),'受測者 A'),),(),())
    boxes=[Box((10,10,30,30),'王',.95),Box((32,10,75,30),'小明',.95)]
    masks=Detector(cfg,NoSpans()).masks(boxes,200,100)
    alias=next(m for m in masks if m.replacement=='受測者 A')
    assert alias.rect[0]<=10 and alias.rect[2]>=75


def test_rules_cover_identifiers_even_without_model_detection() -> None:
    detector=Detector(Config((),('DemoOnly-Secret!',),()),NoSpans())
    for text in ['A123456789','A 123456789','study@example.invalid','DemoOnly-Secret!','臺北市測試區範例路123號']:
        assert detector.masks([Box((10,10,200,40),text,.99)],400,100)


def test_separate_password_value_is_guarded() -> None:
    masks=Detector(Config((),(),()),NoSpans()).masks([Box((10,10,60,30),'Password',.99)],400,200)
    guard=next(m for m in masks if m.reason=='field_guard')
    assert guard.rect[2]==400 and guard.rect[3]>=60


def test_uncertain_ocr_is_covered() -> None:
    assert Detector(Config((),(),()),NoSpans()).masks([Box((10,10,20,20),'unclear',.01)],100,100)


def test_model_span_crossing_boxes_covers_both() -> None:
    class Spans:
        def spans(self,text: str) -> list[tuple[int,int,str]]:
            return [(0,len(text),'private_person')]
    masks=Detector(Config((),(),()),Spans()).masks([Box((0,0,30,20),'Avery',.9),Box((50,0,100,20),'Example',.9)],200,100)
    assert len(masks)==2


def test_duplicate_alias_and_invalid_rect_rejected() -> None:
    with pytest.raises(ValueError):
        parse_config({'schema_version':1,'participants':[{'aliases':['王小明']},{'aliases':['王 小 明']}]})
    with pytest.raises(ValueError):
        parse_config({'schema_version':1,'regions':[{'rect':[.5,0,.2,1]}]})
    assert letter_id(25)=='Z' and letter_id(26)=='AA'


def test_fixed_participant_regions_do_not_depend_on_ocr(tmp_path: Path) -> None:
    source=tmp_path/'source.mp4';create_video(source)
    cfg=parse_config({'schema_version':1,'participants':[{'aliases':['王小明'],'replacement':'受測者 A'}],
                      'regions':[{'kind':'participant','rect':[0,0,1,.5],'replacement':'受測者 A'}]})
    output=tmp_path/'out.mp4'
    report=process_video(source,output,tmp_path/'report.json',cfg,Detector(cfg,NoSpans()),NoBoxes(),'drop')
    assert report['mask_applications_by_reason']['participant_region']==3
    with av.open(str(output)) as video:
        for frame in video.decode(video=0):
            rgb=frame.to_ndarray(format='rgb24')
            assert rgb[2:30,2:62].max()>220 # alias ink present even without OCR
            assert rgb[2:30,2:62].min()<80 # outline stays legible on the blurred background


def test_fixed_participant_region_time_boundaries() -> None:
    first=Region('A','participant',(0,0,1,1),0,1,'受測者 A')
    second=Region('B','participant',(0,0,1,1),1,None,'受測者 B')
    assert first.active(.999) and not first.active(1)
    assert not second.active(.999) and second.active(1)
    with pytest.raises(ValueError):
        parse_config({'schema_version':1,'regions':[{'kind':'participant','rect':[0,0,1,1],'replacement':'Unknown'}]})


def test_single_frame_notification_and_vfr_timestamps(tmp_path: Path) -> None:
    source=tmp_path/'source.mp4';create_video(source)
    destination=tmp_path/'out.mp4';report=tmp_path/'report.json'
    cfg=config()
    result=process_video(source,destination,report,cfg,Detector(cfg,NoSpans()),NoBoxes(),'drop')
    assert result['frames_processed']==3
    with av.open(str(destination)) as output:
        frames=list(output.decode(video=0))
        assert abs(frames[2].time-.35)<.002
        for frame in frames:
            pixels=frame.to_ndarray(format='rgb24')
            assert pixels[52:62,52:62].mean()>150
        with av.open(str(source)) as original:
            original_frames=list(original.decode(video=0))
        before=original_frames[1].to_ndarray(format='rgb24')[:32,:32].astype(float)
        after=frames[1].to_ndarray(format='rgb24')[:32,:32].astype(float)
        edge_before=np.abs(np.diff(before,axis=0)).mean()+np.abs(np.diff(before,axis=1)).mean()
        edge_after=np.abs(np.diff(after,axis=0)).mean()+np.abs(np.diff(after,axis=1)).mean()
        assert edge_after<edge_before*.4
        assert not output.streams.audio
    serialized=report.read_text()
    assert 'raw_text_in_report' in serialized and 'source.mp4' not in serialized
    assert result['mask_style']=='gaussian_blur'
    assert result['blur_uses_original_pixels'] is True
    assert result['irreversible_redaction_guaranteed'] is False


def test_blur_suppresses_details_retains_colors_and_keeps_distant_pixels() -> None:
    rgb=np.full((200,240,3),[190,145,100],dtype=np.uint8)
    for x in range(72,168,4):rgb[65:105,x:x+2]=[20,20,20]
    mask=Mask((65,55,175,115),'notification')
    blurred=paint(rgb,[mask])
    before=rgb[65:105,72:168].astype(float)
    after=blurred[65:105,72:168].astype(float)
    assert np.abs(np.diff(after,axis=1)).mean()<np.abs(np.diff(before,axis=1)).mean()*.1
    assert np.array_equal(blurred[:30],rgb[:30])
    assert np.array_equal(blurred[:,200:],rgb[:,200:])
    color=after.mean(axis=(0,1))
    assert color[0]>color[1]>color[2]
    alternative=np.full_like(rgb,240)
    assert not np.array_equal(paint(alternative,[mask])[55:115,65:175],blurred[55:115,65:175])


def test_blur_overlap_is_order_independent() -> None:
    rgb=np.full((120,160,3),200,dtype=np.uint8)
    rgb[10:95:3,10:135]=[20,30,40]
    first=Mask((20,25,90,65),'notification');second=Mask((60,40,130,90),'rule')
    assert np.array_equal(paint(rgb,[first,second]),paint(rgb,[second,first]))


def test_failure_stops_without_delivering_partial_video(tmp_path: Path) -> None:
    class Broken:
        def read(self,rgb: np.ndarray) -> list[Box]:
            raise RuntimeError('synthetic failure')
    source=tmp_path/'source.mp4';create_video(source)
    output=tmp_path/'out.mp4';report=tmp_path/'report.json';cfg=config()
    with pytest.raises(RuntimeError):
        process_video(source,output,report,cfg,Detector(cfg,NoSpans()),Broken(),'drop')
    assert not output.exists() and not report.exists()
    assert not list(tmp_path.glob('.vpf-*'))


@pytest.mark.parametrize('audio',['keep','drop'])
def test_explicit_audio_policy_and_private_metadata(tmp_path: Path,audio: str) -> None:
    plain=tmp_path/'plain.mp4';create_video(plain)
    source=tmp_path/'with-audio.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin',
                    '-i',str(plain),'-f','lavfi','-i','sine=frequency=440:sample_rate=44100:duration=1',
                    '-map','0:v','-map','1:a','-c:v','copy','-c:a','aac','-shortest',
                    '-metadata','title=Fictional-private-title','-metadata:s:a:0','title=Fictional-speaker',str(source)],check=True)
    destination=tmp_path/'out.mp4';report=tmp_path/'report.json';cfg=config()
    process_video(source,destination,report,cfg,Detector(cfg,NoSpans()),NoBoxes(),audio)
    with av.open(str(destination)) as output:
        assert len(output.streams.audio)==(1 if audio=='keep' else 0)
        metadata=json.dumps({'global':dict(output.metadata),'streams':[dict(s.metadata) for s in output.streams]})
        assert 'Fictional-private-title' not in metadata and 'Fictional-speaker' not in metadata


def test_never_overwrites_existing_result(tmp_path: Path) -> None:
    source=tmp_path/'source.mp4';create_video(source)
    output=tmp_path/'out.mp4';output.write_bytes(b'existing')
    cfg=config()
    with pytest.raises(ValueError):
        process_video(source,output,tmp_path/'report.json',cfg,Detector(cfg,NoSpans()),NoBoxes(),'drop')
    assert output.read_bytes()==b'existing'


def test_email_does_not_absorb_captcha_on_the_next_row() -> None:
    email=Box((20,20,230,45),'alex@example.invalid',.99)
    captcha=Box((20,75,220,100),"I'm not a robot",.99)
    masks=Detector(Config((),(),()),NoSpans()).masks([email,captcha],500,200)
    assert masks
    assert not any(m.rect[0]<captcha.rect[2] and m.rect[2]>captcha.rect[0] and
                   m.rect[1]<captcha.rect[3] and m.rect[3]>captcha.rect[1] for m in masks)


def test_spatially_adjacent_split_email_is_still_protected() -> None:
    boxes=[Box((20,20,65,45),'alex',.99),Box((67,20,80,45),'@',.99),
           Box((82,20,220,45),'example.invalid',.99)]
    masks=Detector(Config((),(),()),NoSpans()).masks(boxes,500,200)
    for box in boxes:
        assert any(m.rect[0]<=box.rect[0] and m.rect[2]>=box.rect[2] for m in masks)


def test_unrelated_rows_do_not_form_an_email() -> None:
    boxes=[Box((20,20,100,45),'alex@',.99),Box((20,75,220,100),'example.invalid',.99)]
    assert Detector(Config((),(),()),NoSpans()).masks(boxes,500,200)==[]
