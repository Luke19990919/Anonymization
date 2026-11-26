from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from .video import load_font


def make_demo(folder: Path) -> dict[str, object]:
    folder.mkdir(parents=True, exist_ok=True)
    video_path = folder/'synthetic-input.mp4'
    config_path = folder/'config.synthetic.json'
    truth_path = folder/'expected.synthetic.json'
    if any(p.exists() for p in [video_path, config_path, truth_path]):
        raise ValueError('示範目錄已含檔案，請選擇新的目錄。')
    config = {
        'schema_version':1,
        'participants':[
            {'aliases':['王小明'], 'replacement':'受測者 A'},
            {'aliases':['陳小華'], 'replacement':'受測者 B'},
        ],
        'sensitive_literals':['study.user@example.invalid'],
        'regions':[
            {'name':'notification', 'kind':'notification', 'rect':[0.60, 0, 1, 0.28]},
            {'name':'password-field', 'kind':'manual', 'rect':[0.02, 0.64, 0.99, 0.80]},
            {'name':'name-A-first', 'kind':'participant', 'rect':[0.03, 0.20, 0.25, 0.30], 'start':0, 'end':1, 'replacement':'受測者 A'},
            {'name':'name-B', 'kind':'participant', 'rect':[0.03, 0.20, 0.25, 0.30], 'start':1, 'end':2, 'replacement':'受測者 B'},
            {'name':'name-A-again', 'kind':'participant', 'rect':[0.03, 0.20, 0.25, 0.30], 'start':2, 'end':None, 'replacement':'受測者 A'},
        ],
        'padding':8, 'low_confidence':0.3, 'languages':['ch_tra', 'en'],
    }
    font = load_font(29)
    small = load_font(24)
    with av.open(str(video_path), 'w') as container:
        stream = container.add_stream('libx264', rate=4)
        stream.width, stream.height, stream.pix_fmt = 960, 640, 'yuv420p'
        stream.options = {'crf':'15'}
        for index in range(12):
            image = Image.new('RGB', (960, 640), '#f4f5f7')
            draw = ImageDraw.Draw(image)
            draw.text((35, 30), '虛構測試資料 / Fictional demo', font=small, fill='black')
            name = '陳小華' if 4 <= index < 8 else '王小明'
            draw.text((40, 140), name, font=font, fill='black')
            draw.text((40, 220), 'study.user@example.invalid', font=font, fill='black')
            draw.text((40, 290), 'A123456789', font=font, fill='black')
            draw.text((40, 360), '臺北市測試區範例路123號', font=font, fill='black')
            if index == 8:
                draw.text((40, 435), 'password: DemoOnly-NotARealSecret-2026!', font=small, fill='black')
            draw.text((40, 515), 'My name is Alice Smith.', font=small, fill='black')
            draw.text((40, 570), '一般操作畫面保留 / Normal content', font=small, fill='black')
            if index == 5:
                draw.rounded_rectangle((600, 30, 945, 165), radius=12, fill='#d6e8ff')
                draw.text((615, 45), '私人通知（虛構）', font=small, fill='black')
                draw.text((615, 90), '不含個資的私密內容', font=small, fill='black')
            frame = av.VideoFrame.from_ndarray(np.array(image), format='rgb24')
            frame.pts, frame.time_base = index, Fraction(1, 4)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    truth = {'synthetic':True, 'frames':12, 'fps':4,
             'single_frame_notification_index':5, 'single_frame_password_index':8,
             'participant_sequence':['A']*4+['B']*4+['A']*4,
             'notification_check_rect':[620, 40, 920, 150],
             'password_check_rect':[40, 435, 900, 480],
             'retained_content_check_rect':[35, 575, 800, 615]}
    truth_path.write_text(json.dumps(truth,indent=2)+'\n',encoding='utf-8')
    return {'video':str(video_path), 'config':str(config_path), 'synthetic':True}
