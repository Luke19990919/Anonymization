"""Locate closed input surfaces below labels, independent of recognized value text."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .detection import Box


def below_label_surface(rgb: np.ndarray,label: Box) -> tuple[int,int,int,int] | None:
    height,width=rgb.shape[:2]
    x1,y1,x2,y2=label.rect
    label_height=max(1,y2-y1)
    top=max(0,y2-4);bottom=min(height,y2+label_height*7)
    if bottom<=top:return None
    search=rgb[top:bottom].astype(np.int16)
    colors:set[tuple[int,...]]=set()
    candidates:set[tuple[int,int,int,int]]=set()
    for offset_y in (.5,1,2,3):
        y=min(bottom-1,y2+int(label_height*offset_y))
        for x in (x1+12,x2+12,x1+int(width*.1)):
            if not 0<=x<width:continue
            color=tuple(int(v) for v in rgb[y,x])
            if color in colors:continue
            colors.add(color)
            similar=np.max(np.abs(search-np.array(color)),axis=2)<=12
            components,_=ndimage.label(similar)
            bounds=ndimage.find_objects(components)
            for region_id,region in enumerate(bounds,1):
                if region is None:continue
                ys,xs=region
                w,h=xs.stop-xs.start,ys.stop-ys.start
                actual_top=top+ys.start
                if ys.start==0 or ys.stop==bottom-top:continue
                if abs(xs.start-x1)>max(24,(x2-x1)*.55):continue
                if not (y2-2<=actual_top<=y2+label_height*2.5):continue
                if not (max(80,(x2-x1)*1.2)<=w<=width*.9 and
                        max(20,label_height*.65)<=h<=label_height*3.5):continue
                if np.mean(components[ys,xs]==region_id)<.65:continue
                candidates.add((xs.start,actual_top,xs.stop,top+ys.stop))
    return min(candidates,key=lambda r:(r[1]-y2,abs(r[0]-x1))) if candidates else None
