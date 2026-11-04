#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recognize_stream.py —— 实时人脸识别（鉴权）

流程（对应开题报告「实时鉴权」）：
  抓帧 -> RetinaFace(框+5点) -> 5点对齐 -> 提特征 -> 与本地人脸库比对
  -> 在检测框上绘制“姓名+分数” -> 保存 recognize_out.jpg

依赖：align.py(检测+对齐) / recognizer.py(提特征) / face_db.py(库)
不依赖 OpenCV，仅 NumPy + Pillow + NCNN。

用法:
  python3 recognize_stream.py          # 无限运行，Ctrl+C 退出
  python3 recognize_stream.py 10       # 运行 10 秒后自动退出
"""
import os
import sys
import io
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import align
import recognizer
import face_db

DB_PATH = os.path.join(HERE, "face_db.npz")
OUT_PATH = os.path.join(HERE, "recognize_out.jpg")
RECOGNIZE_THRESH = 0.4     # 识别阈值：调高降低误识，调低提升召回


def _draw_box(draw, box, label):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    color = (0, 255, 0) if label.lower() != "unknown" else (255, 0, 0)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((x1, max(0, y1 - 18)), label, fill=color, font=font)


def main(duration=0.0):
    rec = recognizer.Recognizer()
    db = face_db.FaceDB(path=DB_PATH)
    if os.path.exists(DB_PATH):
        db.load(DB_PATH)
    print("人脸库人数: %d，识别阈值: %.2f" % (len(db), RECOGNIZE_THRESH))
    if len(db) == 0:
        print("（提示：库为空，所有结果都会是 unknown；先跑 enroll.py 注册）")

    start = time.time()
    frame = 0
    try:
        while True:
            if duration > 0 and (time.time() - start) > duration:
                break
            try:
                data = align.grab_jpeg()
            except RuntimeError as e:
                print("相机错误: %s" % e)
                break

            dets = align.detect_faces_from_jpeg(data)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            draw = ImageDraw.Draw(img)

            if not dets:
                print("[帧 %d] 未检测到人脸" % frame)
            for box, lm in dets:
                aligned = align.warp_to_112(img, lm)           # 5 点对齐
                emb = rec.extract(aligned)
                name, score = db.recognize(emb, threshold=RECOGNIZE_THRESH)
                label = "%s %.2f" % (name, score)
                _draw_box(draw, box, label)
                print("[帧 %d] 识别: %s  分数 %.4f" % (frame, name, score))

            img.save(OUT_PATH)
            frame += 1
            time.sleep(0.05)    # 轻微节流，降低 CPU 占用
    except KeyboardInterrupt:
        print("\n用户中断，退出。")
    print("识别结束，共处理 %d 帧，最后结果存于 %s" % (frame, OUT_PATH))


if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    main(dur)
