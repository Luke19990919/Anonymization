from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, TypeVar
from fractions import Fraction
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from itertools import zip_longest
from typing import Callable, Protocol

import av
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from .config import Config
from .detection import Box, Detector, Mask
from .temporal import ProtectedFrame, NotificationLookahead
from . import __version__

BLUR_RADIUS_MIN = 24
BLUR_RADIUS_MAX = 80
BLUR_FEATHER = 16


class OCR(Protocol):
    def read(self, rgb: np.ndarray) -> list[Box]: ...


_Item = TypeVar('_Item')


def prefetch_one(items: Iterator[_Item]) -> Iterator[_Item]:
    """Overlap one OCR frame with ordered classification; propagate every failure."""
    sentinel = object()
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='vpf-ocr-prefetch') as executor:
        pending = executor.submit(next,items,sentinel)
        try:
            while True:
                item = pending.result()
                if item is sentinel:
                    return
                pending = executor.submit(next,items,sentinel)
                yield item
        finally:
            pending.cancel()


def font_path() -> Path:
    return Path(str(files('video_privacy').joinpath('assets/NotoSansTC.ttf')))


@lru_cache(maxsize=48)
def load_font(size: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(font_path()),size)
    font.set_variation_by_name('Regular')
    return font


def paint(rgb: np.ndarray, masks: list[Mask], *, solid: bool = False) -> np.ndarray:
    image = Image.fromarray(rgb)
    if not masks:
        return np.array(image)
    if solid:
        draw = ImageDraw.Draw(image)
        for mask in masks:
            x1,y1,x2,y2 = mask.rect
            draw.rectangle((x1, y1, x2-1, y2-1), fill=(0, 0, 0))
    else:
        core = Image.new('L', image.size, 0)
        support = Image.new('L', image.size, 0)
        expanded = Image.new('L', image.size, 0)
        core_draw,support_draw = ImageDraw.Draw(core),ImageDraw.Draw(support)
        expanded_draw = ImageDraw.Draw(expanded)
        for mask in masks:
            x1,y1,x2,y2 = mask.rect
            core_draw.rectangle((x1,y1,x2-1,y2-1),fill=255)
            support_draw.rectangle((x1-BLUR_FEATHER,y1-BLUR_FEATHER,
                                    x2-1+BLUR_FEATHER,y2-1+BLUR_FEATHER),fill=255)
            expanded_draw.rectangle((x1-8,y1-8,x2+7,y2+7),fill=255)
        # The sensitive core receives full-strength blur; only the outer margin is feathered.
        feather = expanded.filter(ImageFilter.GaussianBlur(6))
        alpha = ImageChops.lighter(core,ImageChops.darker(feather,support))
        smallest_side = max(min(m.rect[2]-m.rect[0],m.rect[3]-m.rect[1]) for m in masks)
        radius = max(BLUR_RADIUS_MIN,min(BLUR_RADIUS_MAX,smallest_side*0.65))
        image = Image.composite(image.filter(ImageFilter.GaussianBlur(radius)),image,alpha)
    draw = ImageDraw.Draw(image)
    for mask in masks:
        if mask.replacement is None:
            continue
        x1, y1, x2, y2 = mask.rect
        # Shrink the alias to the original masked box; never paint into other content.
        for size in range(min(32, max(8, y2-y1-6)), 5, -1):
            font = load_font(size)
            bounds = draw.textbbox((0, 0), mask.replacement, font=font)
            if bounds[2]-bounds[0] <= x2-x1-6 and bounds[3]-bounds[1] <= y2-y1-4:
                draw.text((x1+3-bounds[0], y1+2-bounds[1]), mask.replacement,
                          fill=(255, 255, 255), font=font,
                          stroke_width=0 if solid else 1, stroke_fill=(10,15,20))
                break
    return np.array(image)


def inspect_video(path: Path) -> dict[str, object]:
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError('此版本只接受恰有一個視訊軌的影片。')
        stream = container.streams.video[0]
        return {'width':stream.width, 'height':stream.height,
                'average_fps':float(stream.average_rate) if stream.average_rate else None,
                'time_base':str(stream.time_base), 'declared_frames':stream.frames,
                'duration_seconds':float(container.duration/av.time_base) if container.duration else None,
                'audio_tracks':len(container.streams.audio), 'video_codec':stream.codec_context.name}


def _mux(redacted: Path, original: Path, destination: Path, audio: str) -> None:
    import imageio_ffmpeg
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
               '-nostdin', '-n', '-copyts', '-i', str(redacted)]
    if audio == 'keep':
        command += ['-i', str(original), '-map', '0:v:0', '-map', '1:a?', '-c:a', 'aac']
    else:
        command += ['-map', '0:v:0', '-an']
    command += ['-c:v', 'copy', '-map_metadata', '-1', '-map_metadata:s', '-1', '-map_chapters', '-1',
                '-sn', '-dn', '-movflags', '+faststart', str(destination)]
    completed = subprocess.run(command, capture_output=True, timeout=None, check=False)
    if completed.returncode != 0:
        # FFmpeg errors can include input metadata; do not copy them into reports/logs.
        raise RuntimeError('影片封裝失敗；請確認檔案可讀且輸出目錄空間足夠。')


def process_video(
    source: Path, destination: Path, report_path: Path, config: Config,
    detector: Detector, ocr: OCR, audio: str,
    progress: Callable[[int], None] | None = None,
    region_observer: Callable[[int,list[Mask]],None] | None = None,
    *, prefetch_ocr: bool = True,
) -> dict[str, object]:
    if audio not in {'keep', 'drop'}:
        raise ValueError('必須明確選擇保留或移除音軌。')
    if destination.suffix.lower() != '.mp4':
        raise ValueError('輸出檔案必須使用 .mp4 副檔名。')
    paths = [source.resolve(), destination.resolve(), report_path.resolve()]
    if len(set(paths)) != len(paths) or destination.exists() or report_path.exists():
        raise ValueError('輸入、影片輸出、報告必須使用不同路徑；不覆寫既有檔案。')
    if not source.is_file():
        raise ValueError('找不到輸入影片。')
    destination.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    timings: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    frames = empty_ocr = reused = 0
    first_time: Fraction | None = None
    last_time: Fraction | None = None
    last_hash: bytes | None = None
    previous_masks: list[Mask] = []
    output_created = report_created = False
    lookahead=NotificationLookahead(getattr(detector,'notification_motion_guard_seconds',0.0))
    # Temporary video contains only already-masked frames, never raw frame exports.
    with tempfile.TemporaryDirectory(prefix='.vpf-', dir=destination.parent) as folder:
        temporary = Path(folder)/'redacted-only.mp4'
        completed_video = Path(folder)/'completed.mp4'
        with av.open(str(source)) as container, av.open(str(temporary), 'w') as output:
            if len(container.streams.video) != 1:
                raise ValueError('此版本只接受恰有一個視訊軌的影片。')
            stream = container.streams.video[0]
            if stream.metadata.get('rotate', '0') not in {'0', '0.0'}:
                raise ValueError('此版本需要無旋轉標記的螢幕錄影，請先轉成正向影片。')
            encoder = output.add_stream('libx264', rate=stream.average_rate or Fraction(30, 1))
            encoder.width = stream.width+(stream.width % 2)
            encoder.height = stream.height+(stream.height % 2)
            encoder.pix_fmt = 'yuv420p'
            encoder.options = {'crf':'18', 'preset':'medium'}
            encoder.time_base = stream.time_base
            encoder.codec_context.time_base = stream.time_base
            def emit(item: ProtectedFrame) -> None:
                nonlocal frames
                rgb=item.rgb
                height,width=rgb.shape[:2]
                fixed,dynamic=item.fixed,item.dynamic
                final_masks=fixed+item.guards
                stage = perf_counter()
                masked = paint(rgb, dynamic)
                # A final pass gives manually marked regions priority over alias overlays.
                masked = paint(masked, final_masks) if final_masks else masked
                if masked.shape[1] != encoder.width or masked.shape[0] != encoder.height:
                    padded = np.zeros((encoder.height, encoder.width, 3), dtype=np.uint8)
                    padded[:height, :width] = masked
                    masked = padded
                timings['mask_render_seconds'] += perf_counter()-stage
                stage = perf_counter()
                new_frame = av.VideoFrame.from_ndarray(masked, format='rgb24')
                new_frame.pts, new_frame.time_base = item.pts, item.time_base
                for packet in encoder.encode(new_frame):
                    output.mux(packet)
                timings['encode_mux_seconds'] += perf_counter()-stage
                for mask in fixed+dynamic+item.guards:
                    reasons[mask.reason] += 1
                if region_observer is not None:
                    region_observer(frames,fixed+dynamic+item.guards)
                frames += 1
                if progress is not None:
                    progress(frames)
            def read_frames() -> Iterator[tuple]:
                nonlocal reused,empty_ocr,first_time,last_time,last_hash
                decoded = iter(container.decode(stream))
                while True:
                    stage = perf_counter()
                    try:
                        frame = next(decoded)
                    except StopIteration:
                        timings['decode_seconds'] += perf_counter()-stage
                        break
                    timings['decode_seconds'] += perf_counter()-stage
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError('影片缺少畫面時間戳，停止處理以避免音畫錯位。')
                    if getattr(frame, 'rotation', 0) != 0:
                        raise ValueError('此版本需要正向影片；不處理帶旋轉標記的畫面。')
                    timestamp = frame.pts*frame.time_base
                    if last_time is not None and timestamp <= last_time:
                        raise ValueError('影片畫面時間戳未嚴格遞增，停止處理。')
                    if first_time is None:
                        first_time = timestamp
                    last_time = timestamp
                    seconds = float(timestamp-first_time)
                    stage = perf_counter()
                    rgb = frame.to_ndarray(format='rgb24')
                    timings['pixel_conversion_seconds'] += perf_counter()-stage
                    stage = perf_counter()
                    height, width = rgb.shape[:2]
                    fixed = [Mask(region.pixels(width, height),
                                  'participant_region' if region.kind == 'participant' else region.kind,
                                  region.replacement)
                             for region in config.regions if region.active(seconds)]
                    # Notifications and manually marked fields are covered before OCR.
                    guarded = paint(rgb, fixed, solid=True) if fixed else rgb
                    digest = hashlib.sha256(guarded.tobytes()).digest()
                    timings['guard_hash_seconds'] += perf_counter()-stage
                    if digest == last_hash:
                        boxes = None
                        reused += 1
                    else:
                        stage = perf_counter()
                        boxes = ocr.read(guarded)
                        timings['ocr_roundtrip_seconds'] += perf_counter()-stage
                        if not boxes:
                            empty_ocr += 1
                        last_hash = digest
                    yield frame,rgb,fixed,guarded,boxes,seconds
            prepared = read_frames()
            ordered = prefetch_one(prepared) if prefetch_ocr else prepared
            try:
                for frame,rgb,fixed,guarded,boxes,seconds in ordered:
                    height,width=rgb.shape[:2]
                    if boxes is None:
                        dynamic = previous_masks
                    else:
                        stage = perf_counter()
                        if hasattr(detector,'set_frame'):
                            detector.set_frame(guarded)
                        dynamic = detector.masks(boxes, width, height)
                        timings['classification_rules_seconds'] += perf_counter()-stage
                        previous_masks = dynamic
                    item=ProtectedFrame(rgb,seconds,frame.pts,frame.time_base,fixed,dynamic)
                    for ready in lookahead.push(item):emit(ready)
            finally:
                ordered.close()
            for ready in lookahead.finish():emit(ready)
            stage = perf_counter()
            for packet in encoder.encode():
                output.mux(packet)
            timings['encode_mux_seconds'] += perf_counter()-stage
        if frames == 0:
            raise ValueError('影片沒有可解碼的畫面。')
        stage = perf_counter()
        _mux(temporary, source, completed_video, audio)
        timings['final_mux_seconds'] += perf_counter()-stage
        stage = perf_counter()
        with av.open(str(source)) as original, av.open(str(completed_video)) as verified:
            if len(verified.streams.audio) != (len(original.streams.audio) if audio == 'keep' else 0):
                raise RuntimeError('音軌輸出與選擇不符，影片未交付。')
            if verified.streams.subtitles or verified.streams.data:
                raise RuntimeError('輸出含有非預期的附加軌，影片未交付。')
            count = 0
            for before, after in zip_longest(original.decode(video=0), verified.decode(video=0)):
                if before is None or after is None:
                    raise RuntimeError('輸出畫面數不符，影片未交付。')
                if before.pts is None or after.pts is None or abs(float(before.pts*before.time_base-after.pts*after.time_base)) > 0.002:
                    raise RuntimeError('輸出時間戳與原片不符，影片未交付。')
                count += 1
            if count != frames:
                raise RuntimeError('輸出畫面數不符，影片未交付。')
        timings['output_verification_seconds'] += perf_counter()-stage
        report: dict[str, object] = {
            'performance': {'stage_seconds': dict(timings), 'processing_seconds': perf_counter()-started,
                            'width': stream.width, 'height': stream.height,
                            'scope': 'process_video only; excludes model initialization and final delivery copy',
                            'nested_stages': 'OCR worker stages and NLP are subsets of parent timings',
                            'ocr_prefetch_enabled': prefetch_ocr,
                            'overlap': 'OCR and decoding may overlap classification/rendering; stage sums may exceed wall time'},
            'schema_version':1, 'program_version':__version__, 'frames_processed':frames,
            'mask_style':'gaussian_blur', 'blur_radius_min_pixels':BLUR_RADIUS_MIN,
            'blur_radius_max_pixels':BLUR_RADIUS_MAX, 'blur_feather_pixels':BLUR_FEATHER,
            'blur_uses_original_pixels':True, 'irreversible_redaction_guaranteed':False,
            'frames_reusing_identical_ocr':reused, 'distinct_frames_without_ocr_boxes':empty_ocr,
            'mask_applications_by_reason':dict(reasons), 'audio_policy':audio,
            'speech_redacted':False, 'notifications':'configured regions only',
            'ocr_languages':list(config.languages), 'frame_sampling':'none; every decoded frame processed',
            'timestamps':'source presentation timestamps preserved',
            'review_required':True, 'raw_text_in_report':False,
        }
        if hasattr(detector,'summary'):
            report.update(detector.summary())
        classifier = getattr(detector,'classifier',None)
        if hasattr(classifier,'inference_seconds'):
            report['nlp_inference_seconds'] = classifier.inference_seconds
            report['nlp_inference_calls'] = classifier.inference_calls
        report.update(lookahead.summary())
        if hasattr(ocr,'summary'):
            report.update(ocr.summary())
        # Exclusive output creation; failure never leaves a partial shareable video.
        try:
            with destination.open('xb') as target, completed_video.open('rb') as ready:
                output_created = True
                while chunk := ready.read(8*1024*1024):
                    target.write(chunk)
            with report_path.open('x', encoding='utf-8') as target:
                report_created = True
                json.dump(report, target, ensure_ascii=False, indent=2)
                target.write('\n')
        except BaseException:
            if output_created:
                destination.unlink(missing_ok=True)
            if report_created:
                report_path.unlink(missing_ok=True)
            raise
        return report
