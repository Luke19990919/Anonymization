"""Generate a fictional branded notification with a single-frame appearance."""
from pathlib import Path
from fractions import Fraction
import av,json,numpy as np
import sys
from PIL import Image,ImageDraw
from video_privacy.video import load_font
root=Path(sys.argv[1])
root.mkdir(exist_ok=True,mode=0o700)
gold=[]
with av.open(str(root/'input.mp4'),'w') as out:
 stream=out.add_stream('libx264',rate=10);stream.width,stream.height,stream.pix_fmt=800,600,'yuv420p'
 for index in range(24):
  image=Image.new('RGB',(800,600),'#ffffff');draw=ImageDraw.Draw(image)
  draw.text((30,40),'一般操作範例',font=load_font(24),fill='black')
  targets=[]
  if index in (5,6,14):
   draw.rectangle((410,120,779,269),fill='#dceaff',outline='#6b8ec2',width=2)
   draw.text((435,135),'LINE',font=load_font(25),fill='#102030')
   draw.text((435,190),'今晚的安排',font=load_font(23),fill='#102030')
   targets=[(410,120,780,270)]
  gold.append({'frame':index,'targets':targets})
  f=av.VideoFrame.from_ndarray(np.array(image),format='rgb24');f.pts,f.time_base=index,Fraction(1,10)
  for p in stream.encode(f):out.mux(p)
 for p in stream.encode():out.mux(p)
(root/'gold.json').write_text(json.dumps({'synthetic':True,'frames':gold},indent=2)+'\n')
