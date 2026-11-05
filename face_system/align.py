#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align.py —— 5 点人脸对齐模块（开题报告研究内容②外扩+仿射变换的增强版）

依赖：仅 NumPy + Pillow + NCNN（与现有约束一致，不依赖 OpenCV）。
功能：
  1) 用 RetinaFace-mobile0.25（单 NCNN 模型）同时输出 检测框 + 5 个关键点；
  2) 对 5 关键点做“几何重排”，自动判定 左眼/右眼/鼻/左嘴/右嘴，避免魔改模型输出顺序导致的镜像；
  3) 用 Umeyama 相似变换把 5 点 warp 到 ArcFace 标准 112x112 模板，供 MobileFaceNet 提特征。

注意：本文件只是“检测+对齐”，特征提取仍在 recognizer.py，比对在 face_db.py。
"""
import os
import sys
import subprocess
import numpy as np
from io import BytesIO
from PIL import Image
import ncnn

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RETINA_DIR = os.path.join(HERE, "models", "retinaface")
RETINA_PARAM = os.path.join(RETINA_DIR, "face.param")
RETINA_BIN = os.path.join(RETINA_DIR, "face.bin")

CONF_THRESH = 0.6      # 人脸置信阈值（与模型训练一致）
NMS_THRESH = 0.4       # NMS IoU 阈值

# RetinaFace-mobile0.25 的 anchor 配置（4 个 stride）
_STEPS = [8, 16, 32, 64]
_MINS = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]

# ArcFace / InsightFace 标准 112x112 五点模板
# 顺序：[左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]（图像左=较小 x）
TEMPLATE = np.array([
    [38.2946, 51.6963],   # 左眼
    [73.5318, 51.5014],   # 右眼
    [56.0252, 71.7366],   # 鼻尖
    [41.5493, 92.3655],   # 左嘴角
    [70.7299, 92.2041],   # 右嘴角
], dtype=np.float64)

_CAM_W, _CAM_H = 640, 360
_CAM_DEV = "/dev/video0"


class _RetinaDetector:
    """RetinaFace-mobile0.25 单例，进程内只加载一次。"""

    _inst = None

    def __init__(self):
        if not (os.path.exists(RETINA_PARAM) and os.path.exists(RETINA_BIN)):
            raise FileNotFoundError(
                "RetinaFace 模型缺失：%s / %s" % (RETINA_PARAM, RETINA_BIN))
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = 2
        ret = self.net.load_param(RETINA_PARAM)
        if ret != 0:
            raise RuntimeError("RetinaFace 参数加载失败: %s (code=%d)" % (RETINA_PARAM, ret))
        ret = self.net.load_model(RETINA_BIN)
        if ret != 0:
            raise RuntimeError("RetinaFace 权重加载失败: %s (code=%d)" % (RETINA_BIN, ret))

    @classmethod
    def instance(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst


# ----------------------------------------------------------------------------
# 检测：图像(PIL RGB) -> [(box, landmarks), ...]
#   box = (x1, y1, x2, y2, score)
#   landmarks = (N,2) 原图像素坐标，顺序为模型原始输出（后续 geometric_reorder 规整）
# ----------------------------------------------------------------------------
def detect_faces_from_image(img):
    """输入 PIL RGB 图像，返回检测列表。"""
    W, H = img.size
    bgr = np.asarray(img)[:, :, ::-1].copy()           # RGB -> BGR
    mat = ncnn.Mat.from_pixels(bgr, ncnn.Mat.PixelType.PIXEL_BGR, W, H)
    mat.substract_mean_normalize([104.0, 117.0, 123.0], [1.0, 1.0, 1.0])

    det = _RetinaDetector.instance()
    ex = det.net.create_extractor()
    ex.input(0, mat)
    r0, out0 = ex.extract("output0")
    r1, out1 = ex.extract("530")
    r2, out2 = ex.extract("529")
    if r0 != 0 or r1 != 0 or r2 != 0:
        raise RuntimeError("RetinaFace 输出提取失败: %d, %d, %d" % (r0, r1, r2))

    bbox = np.array(out0, dtype=np.float64).reshape(-1, 4)
    score = np.array(out1, dtype=np.float64).reshape(-1, 2)
    lmk = np.array(out2, dtype=np.float64).reshape(-1, 10)

    N = bbox.shape[0]
    # ---- 构建与模型完全一致的 prior boxes（归一化坐标）----
    anc = []
    for k in range(4):
        fh = int(np.ceil(H / _STEPS[k]))
        fw = int(np.ceil(W / _STEPS[k]))
        for i in range(fh):
            for j in range(fw):
                for ms in _MINS[k]:
                    anc.append([(j + 0.5) * _STEPS[k] / W,
                                (i + 0.5) * _STEPS[k] / H,
                                ms / W, ms / H])
    anc = np.array(anc, dtype=np.float64)
    assert anc.shape[0] == N, "anchor 数量与输出不匹配: %d vs %d" % (anc.shape[0], N)

    conf = score[:, 1]                  # 取人脸类
    # ---- 解码 bbox ----
    cx = anc[:, 0] + bbox[:, 0] * 0.1 * anc[:, 2]
    cy = anc[:, 1] + bbox[:, 1] * 0.1 * anc[:, 3]
    sw = anc[:, 2] * np.exp(bbox[:, 2] * 0.2)
    sh = anc[:, 3] * np.exp(bbox[:, 3] * 0.2)
    x1 = np.clip((cx - sw / 2) * W, 0, W)
    y1 = np.clip((cy - sh / 2) * H, 0, H)
    x2 = np.clip((cx + sw / 2) * W, 0, W)
    y2 = np.clip((cy + sh / 2) * H, 0, H)

    # ---- 解码 landmarks（已在原图像素坐标）----
    lms = np.zeros((N, 5, 2), dtype=np.float64)
    for j in range(5):
        lms[:, j, 0] = (anc[:, 0] + lmk[:, j * 2] * 0.1 * anc[:, 2]) * W
        lms[:, j, 1] = (anc[:, 1] + lmk[:, j * 2 + 1] * 0.1 * anc[:, 3]) * H

    # ---- 阈值 + NMS ----
    mask = conf > CONF_THRESH
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    boxes = np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1)
    scores = conf[mask]
    lm_out = lms[mask]
    keep = _nms(boxes, scores, NMS_THRESH)

    results = []
    for i in keep:
        box = (float(boxes[i, 0]), float(boxes[i, 1]),
               float(boxes[i, 2]), float(boxes[i, 3]), float(scores[i]))
        results.append((box, lm_out[i]))
    return results


def detect_faces_from_jpeg(data):
    """输入 JPEG 字节，返回检测列表。"""
    img = Image.open(BytesIO(data)).convert("RGB")
    return detect_faces_from_image(img)


# ----------------------------------------------------------------------------
# 工具：NMS
# ----------------------------------------------------------------------------
def _nms(boxes, scores, iou_thresh):
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_r - inter
        iou = inter / np.maximum(union, 1e-8)
        order = rest[iou < iou_thresh]
    return keep


# ----------------------------------------------------------------------------
# 几何重排：把任意模型输出顺序规整为 [左眼, 右眼, 鼻, 左嘴, 右嘴]
#   依据：眼睛= y 最小的两个点；嘴= y 最大的两个点；鼻= 中间点；
#        每对里 x 较小的为“左”。
# ----------------------------------------------------------------------------
def geometric_reorder(landmarks):
    pts = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    order = np.argsort(pts[:, 1])                 # 按 y 升序
    eyes = order[:2]
    mouth = order[3:]
    nose = order[2]
    # 眼对：x 小=左眼
    if pts[eyes[0], 0] < pts[eyes[1], 0]:
        left_eye, right_eye = pts[eyes[0]], pts[eyes[1]]
    else:
        left_eye, right_eye = pts[eyes[1]], pts[eyes[0]]
    # 嘴对：x 小=左嘴
    if pts[mouth[0], 0] < pts[mouth[1], 0]:
        left_mouth, right_mouth = pts[mouth[0]], pts[mouth[1]]
    else:
        left_mouth, right_mouth = pts[mouth[1]], pts[mouth[0]]
    return np.array([left_eye, right_eye, pts[nose], left_mouth, right_mouth],
                    dtype=np.float64)


# ----------------------------------------------------------------------------
# Umeyama 相似变换（source -> target），返回 (c, R, t)
# ----------------------------------------------------------------------------
def _umeyama(src, dst):
    src = np.asarray(src, dtype=np.float64)   # (N,2)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    s0 = src - mu_s
    d0 = dst - mu_d
    Sigma = (d0.T @ s0) / n                      # 2x2
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    var_s = np.trace(s0.T @ s0) / n
    c = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - c * (R @ mu_s)
    return c, R, t


# ----------------------------------------------------------------------------
# 对齐：把人脸 warp 到 112x112 标准模板
# ----------------------------------------------------------------------------
def warp_to_112(img, landmarks):
    """img: PIL RGB；landmarks: (5,2) 任意顺序。返回 112x112 PIL RGB 对齐人脸。"""
    src = geometric_reorder(landmarks)        # (5,2) 规整顺序
    c, R, t = _umeyama(src, TEMPLATE)         # 相似变换：template = c*R*src + t

    # 求逆，得到 PIL 需要的 dst->src 仿射：src = A*dst + b
    A = (1.0 / c) * R.T
    b = -(1.0 / c) * (R.T @ t)
    matrix = (A[0, 0], A[0, 1], b[0],
              A[1, 0], A[1, 1], b[1])

    aligned = img.transform((112, 112), Image.AFFINE, matrix,
                            resample=Image.BILINEAR)
    return aligned


# ----------------------------------------------------------------------------
# 便捷：选最大/最置信的一张脸
# ----------------------------------------------------------------------------
def largest_face(detections):
    """detections: list of (box, landmarks)。返回 (box, landmarks) 或 None。"""
    if not detections:
        return None
    best = max(detections, key=lambda d: (d[0][4], (d[0][2] - d[0][0]) * (d[0][3] - d[0][1])))
    return best


# ----------------------------------------------------------------------------
# 相机抓取（v4l2-ctl，与原有实现一致）
# ----------------------------------------------------------------------------
def grab_jpeg(width=_CAM_W, height=_CAM_H, device=_CAM_DEV):
    """抓一帧 JPEG 字节。失败抛出 RuntimeError。"""
    tmp = "/tmp/_cam_frame.jpg"
    cmd = ("v4l2-ctl -d %s --set-fmt-video=width=%d,height=%d,pixelformat=MJPG "
           "--stream-mmap --stream-count=1 --stream-to=%s" %
           (device, width, height, tmp))
    rc = subprocess.run(cmd, shell=True, capture_output=True)
    if rc.returncode != 0:
        raise RuntimeError("v4l2-ctl 抓取失败: %s" % rc.stderr.decode(errors="ignore"))
    with open(tmp, "rb") as f:
        data = f.read()
    if len(data) < 100:
        raise RuntimeError("相机帧异常（可能未接相机或无信号）")
    return data


if __name__ == "__main__":
    # 简易自测：用 test_frame.jpg 验证检测+对齐
    import glob
    candidates = [
        os.path.join(HERE, "test_frame.jpg"),
        "/home/pi4/test_frame.jpg",
    ]
    src = next((p for p in candidates if os.path.exists(p)), None)
    if src is None:
        print("NO_TEST_IMAGE")
    else:
        im = Image.open(src).convert("RGB")
        dets = detect_faces_from_image(im)
        print("检测到人脸数:", len(dets))
        for box, lm in dets:
            print("  box=%.0f,%.0f,%.0f,%.0f conf=%.2f" % (box[0], box[1], box[2], box[3], box[4]))
            print("  原始5点=", np.round(lm, 1).tolist())
            reorder = geometric_reorder(lm)
            print("  规整5点=", np.round(reorder, 1).tolist())
            aligned = warp_to_112(im, lm)
            print("  对齐图尺寸:", aligned.size)
        print("ALIGN_OK")
