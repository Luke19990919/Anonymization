"""Bounded, worker-local memoization for pinned EasyOCR 1.7.2 crop recognition.

Only byte-identical rectified crops with identical recognition parameters match.
Current geometry is always returned; images and text never leave this process.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Any,Callable

import numpy as np


class ExactCropCache:
    def __init__(self,recognize: Callable[...,Any],capacity: int=512) -> None:
        self.recognize=recognize
        self.capacity=capacity
        self.cache: OrderedDict[tuple[Any,...],tuple[str,float]]=OrderedDict()
        self.hits=0
        self.misses=0

    def __call__(self,*args: Any,**kwargs: Any) -> Any:
        # EasyOCR's CPU non-rotation path sends one rectified crop per call.
        # Unknown signatures/batches use the original implementation unchanged.
        if kwargs or len(args)<6 or len(args[5])!=1:
            return self.recognize(*args,**kwargs)
        coordinates,crop=args[5][0]
        pixels=np.ascontiguousarray(crop)
        parameters=args[:3]+args[6:]
        key=(id(args[3]),id(args[4]))+parameters+(pixels.shape,pixels.dtype.str,hashlib.sha256(pixels.tobytes()).digest())
        try:hash(key)
        except TypeError:return self.recognize(*args,**kwargs)
        if key in self.cache:
            self.hits+=1
            self.cache.move_to_end(key)
            text,confidence=self.cache[key]
            return [(coordinates,text,confidence)]
        self.misses+=1
        result=self.recognize(*args,**kwargs)
        if len(result)==1:
            self.cache[key]=(result[0][1],result[0][2])
            if len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return result
