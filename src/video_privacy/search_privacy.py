"""Pixel-validated search-field context. No fixed screen coordinates or user roster."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from .detection import Box

SEARCH_LABELS={'搜尋','搜索','search'}


def enclosing_surface(rgb: np.ndarray,box: Box) -> tuple[int,int,int,int] | None:
    height,width=rgb.shape[:2]
    x1,y1,x2,y2=box.rect
    top=max(0,y1-3*(y2-y1));bottom=min(height,y2+3*(y2-y1))
    if bottom<=top or y1<3:return None
    seed_y=y1-3
    seed_x=max(0,min(width-1,(x1+x2)//2))
    color=rgb[seed_y,seed_x].astype(np.int16)
    similar=np.max(np.abs(rgb[top:bottom].astype(np.int16)-color),axis=2)<=4
    labels,_=ndimage.label(similar)
    identity=int(labels[seed_y-top,seed_x])
    if not identity:return None
    region=ndimage.find_objects(labels,max_label=identity)[identity-1]
    if region is None:return None
    ys,xs=region;w=xs.stop-xs.start;h=ys.stop-ys.start
    if not (max(width*.15,2*(x2-x1))<=w<width*.85 and
            y2-y1<=h<=4*(y2-y1) and ys.start>0 and ys.stop<bottom-top):return None
    if np.mean(labels[ys,xs]==identity)<.65:return None
    return xs.start,ys.start+top,xs.stop,ys.stop+top


class SearchContext:
    def __init__(self) -> None:
        self.surface: tuple[int,int,int,int] | None=None

    def locate(self,rgb: np.ndarray,boxes: list[Box]) -> tuple[int,int,int,int] | None:
        candidates=[b for b in boxes if b.text.strip().lower() in SEARCH_LABELS]
        for box in candidates:
            found=enclosing_surface(rgb,box)
            if found is not None:
                self.surface=found
                return found
        if self.surface is None:return None
        x1,y1,x2,y2=self.surface
        if x2>rgb.shape[1] or y2>rgb.shape[0]:
            self.surface=None
            return None
        # Revalidate a closed flat surface instead of trusting previous geometry.
        found=None
        for ratio in (.8,.5,.05):
            left=x1+int((x2-x1)*ratio)
            probe=Box((left,y1+8,min(x2-8,left+max(20,(x2-x1)//10)),y2-8),'',1.0)
            found=enclosing_surface(rgb,probe)
            if found is not None:break
        overlap=0.0
        if found is not None:
            a,b,c,d=found
            intersection=max(0,min(c,x2)-max(a,x1))*max(0,min(d,y2)-max(b,y1))
            overlap=intersection/max((c-a)*(d-b),(x2-x1)*(y2-y1))
        if found is None or overlap<.7:
            self.surface=None
            return None
        self.surface=found
        return found


def suggestion_surface(rgb: np.ndarray,surface: tuple[int,int,int,int]) -> tuple[int,int,int,int] | None:
    height,width=rgb.shape[:2]
    x1,y1,x2,y2=surface
    top=max(0,y2-4);bottom=min(height,y2+int(height*.8))
    left=max(0,x1-40);right=min(width,x2+40)
    seed_x=min(width-1,x1+12)
    # Rounded corners and gaps can precede the menu; a single seed can hit the page.
    tried=set()
    for offset in (12,(y2-y1)//2,y2-y1):
        seed_y=min(height-1,y2+offset)
        if not top<=seed_y<bottom:continue
        color=tuple(int(v) for v in rgb[seed_y,seed_x])
        if color in tried:continue
        tried.add(color)
        similar=np.max(np.abs(rgb[top:bottom,left:right].astype(np.int16)-np.array(color)),axis=2)<=4
        labels,_=ndimage.label(similar)
        identity=int(labels[seed_y-top,seed_x-left])
        if not identity:continue
        region=ndimage.find_objects(labels,max_label=identity)[identity-1]
        if region is None:continue
        ys,xs=region
        if xs.start==0 or xs.stop==right-left or ys.stop==bottom-top:continue
        if ys.start>1.5*(y2-y1):continue
        if not (.6*(x2-x1)<=xs.stop-xs.start<=1.2*(x2-x1) and ys.stop-ys.start>=y2-y1):continue
        if np.mean(labels[ys,xs]==identity)<.5:continue
        return xs.start+left,ys.start+top,xs.stop+left,ys.stop+top
    return None


def inside(box: Box,rect: tuple[int,int,int,int]) -> bool:
    x=(box.rect[0]+box.rect[2])/2;y=(box.rect[1]+box.rect[3])/2
    return rect[0]<=x<=rect[2] and rect[1]<=y<=rect[3]


def aligned_suggestion_region(boxes: list[Box],surface: tuple[int,int,int,int],height: int) -> tuple[int,int,int,int] | None:
    """Fallback when dark menu and page colors connect; require repeated search icons."""
    import re
    x1,y1,x2,y2=surface
    field_height=y2-y1
    icons=sorted([box for box in boxes if x1<=box.rect[0]<=x1+(x2-x1)*.2
                  and y2<=box.rect[1]<=min(height,y2+height*.75)
                  and re.match(r'^(?:[Qq](?:\s|[\u3400-\u9fff]|$)|[Oo○◯⌕🔍](?:\s|$))',box.text)],
                 key=lambda b:b.rect[1])
    if len(icons)<3 or icons[0].rect[1]-y2>field_height*3:return None
    rows=[]
    for box in icons:
        if not rows or box.rect[1]-rows[-1].rect[1]>field_height*.35:
            rows.append(box)
    if len(rows)<3:return None
    gaps=[b.rect[1]-a.rect[1] for a,b in zip(rows,rows[1:])]
    if not all(field_height*.5<=gap<=field_height*1.8 for gap in gaps):return None
    return x1,y2,x2,min(height,int(rows[-1].rect[3]+field_height*.4))
