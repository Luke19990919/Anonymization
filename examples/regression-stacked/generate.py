"""Create a fictional progressive phone/email fixture; gold stays outside detector input."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image,ImageDraw
from video_privacy.video import load_font


def generate(folder: Path) -> None:
    folder.mkdir(parents=True,exist_ok=True)
    phone='0900000000'
    email='alex@example.invalid'
    cases=[('', '')]+[(phone[:i],'') for i in range(1,len(phone)+1)]+[(phone,email[:i]) for i in range(1,len(email)+1)]+[('', '')]
    gold=[]
    with av.open(str(folder/'input.mp4'),'w') as output:
        stream=output.add_stream('libx264',rate=10)
        stream.width,stream.height,stream.pix_fmt=900,600,'yuv420p'
        for index,(phone_value,email_value) in enumerate(cases):
            image=Image.new('RGB',(900,600),'#fafafa');draw=ImageDraw.Draw(image)
            draw.text((220,35),'虛構資料輸入測試',font=load_font(26),fill='#202020')
            targets=[]
            for label,y,value in [('手機號碼 *',100,phone_value),('電子信箱 *',250,email_value)]:
                draw.text((230,y),label,font=load_font(24),fill='#111111')
                draw.rectangle((230,y+42,730,y+102),fill='white',outline='#888888',width=2)
                if value:
                    xy=(245,y+54)
                    draw.text(xy,value,font=load_font(25),fill='#111111')
                    targets.append(draw.textbbox(xy,value,font=load_font(25)))
            draw.text((230,470),'一般操作說明',font=load_font(24),fill='#111111')
            frame=av.VideoFrame.from_ndarray(np.array(image),format='rgb24')
            frame.pts,frame.time_base=index,Fraction(1,10)
            for packet in stream.encode(frame):output.mux(packet)
            gold.append({'frame':index,'targets':targets})
        for packet in stream.encode():output.mux(packet)
    (folder/'gold.json').write_text(json.dumps({'synthetic':True,'frames':gold},indent=2)+'\n')


if __name__=='__main__':
    generate(Path(sys.argv[1]))
