#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enroll.py —— 人脸注册（录入）

流程（对应开题报告三级级联的前半 + 5 点对齐）：
  抓帧 -> RetinaFace 检测(框+5点) -> 相似变换对齐到112x112 -> MobileFaceNet 提特征
  -> 多帧平均 -> 写入本地人脸库 face_db.npz

依赖：align.py(检测+对齐) / recognizer.py(提特征) / face_db.py(库)
不依赖 OpenCV，仅 NumPy + Pillow + NCNN。
"""
import os
import sys
import io
import time
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import align
import recognizer
import face_db

DB_PATH = os.path.join(HERE, "face_db.npz")


def enroll(name, num_frames=5):
    """从相机采集 num_frames 帧，对齐后平均提特征，注册为 name。"""
    if not name:
        print("ERROR: 必须提供姓名")
        return False

    rec = recognizer.Recognizer()
    db = face_db.FaceDB(path=DB_PATH)
    if os.path.exists(DB_PATH):
        db.load(DB_PATH)

    embs = []
    for i in range(num_frames):
        try:
            data = align.grab_jpeg()
        except RuntimeError as e:
            print("相机错误（跳过）: %s" % e)
            continue
        dets = align.detect_faces_from_jpeg(data)
        f = align.largest_face(dets)
        if f is None:
            print("帧 %d: 未检测到人脸，跳过" % (i + 1))
            continue
        img = Image.open(io.BytesIO(data)).convert("RGB")
        aligned = align.warp_to_112(img, f[1])          # 5 点对齐 -> 112x112
        emb = rec.extract(aligned)
        embs.append(emb)
        print("帧 %d: 对齐提取完成，模长 %.4f" % (i + 1, float(np.linalg.norm(emb))))

    if not embs:
        print("ERROR: 没有成功采集到任何人脸，未注册。")
        return False

    avg = np.mean(embs, axis=0).astype(np.float32)
    avg = face_db.FaceDB.l2_normalize(avg)
    db.enroll(name, avg)
    db.save(DB_PATH)
    print("已注册 [%s]，当前库共 %d 人，代表向量模长 %.4f"
          % (name, len(db), float(np.linalg.norm(avg))))
    return True


def enroll_from_jpeg(data, name, num_frames=1):
    """无相机的离线录入（用于测试）：直接对给定 JPEG 字节注册。"""
    rec = recognizer.Recognizer()
    db = face_db.FaceDB(path=DB_PATH)
    if os.path.exists(DB_PATH):
        db.load(DB_PATH)
    dets = align.detect_faces_from_jpeg(data)
    f = align.largest_face(dets)
    if f is None:
        print("ERROR: 该图未检测到人脸")
        return False
    img = Image.open(io.BytesIO(data)).convert("RGB")
    aligned = align.warp_to_112(img, f[1])
    emb = rec.extract(aligned)
    db.enroll(name, emb)
    db.save(DB_PATH)
    print("已注册 [%s]（离线），当前库共 %d 人" % (name, len(db)))
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 enroll.py <姓名> [帧数=5]")
        print("示例: python3 enroll.py 张三")
        sys.exit(1)
    nm = sys.argv[1]
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    enroll(nm, nf)
