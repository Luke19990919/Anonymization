"""Bounded in-memory look-ahead for right-edge notification animations."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass,field
from fractions import Fraction
import numpy as np
from .detection import Mask


@dataclass
class ProtectedFrame:
    rgb: np.ndarray
    timestamp: float
    pts: int
    time_base: Fraction
    fixed: list[Mask]
    dynamic: list[Mask]
    guards: list[Mask]=field(default_factory=list)


def merge_corridors(masks: list[Mask]) -> list[Mask]:
    result: list[Mask]=[]
    for item in masks:
        rect=item.rect
        visibility_rect=item.visibility_rect
        visibility_colors=item.visibility_colors
        i=0
        while i<len(result):
            other=result[i].rect
            if min(rect[3],other[3])>=max(rect[1],other[1]):
                rect=(min(rect[0],other[0]),min(rect[1],other[1]),max(rect[2],other[2]),max(rect[3],other[3]))
                previous=result[i]
                if (visibility_rect is not None and visibility_rect==previous.visibility_rect
                        and visibility_colors is not None and previous.visibility_colors is not None):
                    visibility_colors=tuple(dict.fromkeys(visibility_colors+previous.visibility_colors))
                else:
                    visibility_rect=None;visibility_colors=None
                result.pop(i);i=0
            else:i+=1
        result.append(Mask(rect,'notification_motion_guard',visibility_rect=visibility_rect,
                           visibility_colors=visibility_colors))
    return result


class NotificationLookahead:
    def __init__(self,seconds: float=0.0,max_bytes: int=1536*1024*1024) -> None:
        if seconds<0:raise ValueError('Notification protection window must be nonnegative')
        self.seconds=seconds
        self.max_bytes=max_bytes
        self.pending: deque[ProtectedFrame]=deque()
        self.events: deque[tuple[float,Mask]]=deque()
        self.bytes=0
        self.peak_bytes=0
        self.protected_frames=0
        self.uniform_color_guards_skipped=0
        self.absent_card_guards_skipped=0

    def _corridors(self,item: ProtectedFrame) -> list[Mask]:
        if self.seconds==0:return []
        h,w=item.rgb.shape[:2];result=[]
        for mask in item.dynamic:
            x1,y1,x2,y2=mask.rect
            if mask.reason=='auto_notification' and (x1+x2)/2>=w*.6 and x2>=w*.8:
                result.append(Mask((max(0,x1-8),max(0,y1-8),w,min(h,y2+8)),'notification_motion_guard',
                                   visibility_rect=mask.visibility_rect,visibility_colors=mask.visibility_colors))
        return merge_corridors(result)

    def _visible_guards(self,item: ProtectedFrame,masks: list[Mask]) -> list[Mask]:
        result=[]
        for mask in masks:
            x1,y1,x2,y2=mask.rect
            patch=item.rgb[y1:y2,x1:x2]
            # Exact uniform-color pixels contain no visible text. Without verified
            # card geometry, even a single changed pixel retains protection.
            if patch.size and np.all(patch==patch[0,0]):
                self.uniform_color_guards_skipped+=1
            elif mask.visibility_rect is not None and mask.visibility_colors:
                a,b,c,d=mask.visibility_rect
                core=item.rgb[b:d,a:c]
                if not core.size or not np.all(core==core[0,0]):
                    result.append(mask)
                    continue
                # The visible card body is separate from blur padding. Preserve
                # partial horizontal entry/exit when its body or border remains
                # anywhere in the corridor. Unknown geometry keeps the old guard.
                pixels=patch.astype(np.int16)
                card_present=any(np.any(np.max(np.abs(pixels-np.asarray(color,dtype=np.int16)),axis=2)<=18)
                                 for color in mask.visibility_colors)
                if card_present:
                    result.append(mask)
                else:
                    self.absent_card_guards_skipped+=1
            else:
                result.append(mask)
        return result

    def _pop(self) -> ProtectedFrame:
        item=self.pending.popleft();self.bytes-=item.rgb.nbytes
        if item.guards:self.protected_frames+=1
        return item

    def push(self,item: ProtectedFrame) -> list[ProtectedFrame]:
        if self.bytes+item.rgb.nbytes>self.max_bytes:
            raise RuntimeError('動態通知保護的記憶體暫存容量不足；已停止輸出，未略過畫面。')
        while self.events and item.timestamp-self.events[0][0]>self.seconds+1e-9:
            self.events.popleft()
        item.guards=self._visible_guards(item,merge_corridors([mask for _,mask in self.events]))
        new=self._corridors(item)
        if new:
            item.guards=self._visible_guards(item,merge_corridors(item.guards+new))
            for previous in self.pending:
                if item.timestamp-previous.timestamp<=self.seconds+1e-9:
                    previous.guards=self._visible_guards(previous,merge_corridors(previous.guards+new))
            for mask in new:self.events.append((item.timestamp,mask))
        self.pending.append(item);self.bytes+=item.rgb.nbytes
        self.peak_bytes=max(self.peak_bytes,self.bytes)
        ready=[]
        # Apply current detections before releasing a frame exactly at the window boundary.
        while self.pending and self.pending[0].timestamp<=item.timestamp-self.seconds+1e-9:
            ready.append(self._pop())
        return ready

    def finish(self) -> list[ProtectedFrame]:
        ready=[]
        while self.pending:ready.append(self._pop())
        self.events.clear()
        return ready

    def summary(self) -> dict[str,object]:
        return {'notification_temporal_protection':{'enabled':self.seconds>0,
                'direction':'right_edge_horizontal_corridor','lookback_seconds':self.seconds,
                'linger_seconds':self.seconds,'guarded_frames':self.protected_frames,
                'uniform_color_guards_skipped':self.uniform_color_guards_skipped,
                'absent_card_guards_skipped':self.absent_card_guards_skipped,
                'peak_raw_frame_buffer_bytes':self.peak_bytes,'raw_frames_spooled_to_disk':False,
                'scope':'recognized right-edge notifications only; short entry/exit; may cover background before and after'}}
