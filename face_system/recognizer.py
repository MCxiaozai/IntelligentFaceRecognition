#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recognizer.py —— 基于 NCNN MobileFaceNet(ArcFace) 的人脸特征提取（无 OpenCV）

对应开题报告「研究内容③（特征提取）」与「技术路线(3)」：提取特征向量。
  - 模型：models/facenet/mobilefacenet.param/.bin（InsightFace mobileFaceNet，128 维）
  - 输入：112x112 RGB
  - 前处理：mean=[127.5,127.5,127.5], norm=[1/128,1/128,1/128]  （(x-127.5)/128）
  - 输出：'fc1' 层 128 维，推理后做 L2 归一化
  - 预处理仅用 Pillow + NumPy，替代 cv2 流水线（符合去 OpenCV 要求）

说明：该模型在“对齐后”的人脸上训练，若不做 5 点对齐，精度会略降；
      align=True 的扩展位已预留（需关键点，可后续接 MTCNN/RetinaFace）。
      若实测识别率异常偏低，把 PIXEL_TYPE 改为 PIXEL_BGR 再试（部分 ncnn 移植用 BGR）。
"""
import os
import numpy as np
from PIL import Image
import ncnn


class Recognizer:
    INPUT_SIZE = 112
    MEAN = [127.5, 127.5, 127.5]
    NORM = [1.0 / 128.0, 1.0 / 128.0, 1.0 / 128.0]
    INPUT_BLOB = "data"
    OUTPUT_BLOB = "fc1"
    PIXEL_TYPE = ncnn.Mat.PixelType.PIXEL_RGB   # 如精度异常偏低可改 PIXEL_BGR

    # 注意：本机 ncnn 1.0 在 num_threads>1 时对该模型存在线程竞态 -> 不稳定段错误，
    # 故识别默认单线程（MobileFaceNet 极小，单线程足够快且稳定）。
    def __init__(self, model_dir="models/facenet", num_threads=1):
        self.model_dir = model_dir
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = num_threads
        param = os.path.join(model_dir, "mobilefacenet.param")
        binf = os.path.join(model_dir, "mobilefacenet.bin")
        self.net.load_param(param)
        self.net.load_model(binf)

    def _preprocess(self, face_img):
        """face_img: PIL.Image(RGB)，任意尺寸 → 112x112 RGB numpy(uint8)"""
        img = face_img.convert("RGB").resize((self.INPUT_SIZE, self.INPUT_SIZE),
                                             Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    def extract(self, face_img):
        """输入 PIL.Image(RGB)（裁剪出的人脸），返回 128 维 L2 归一化向量(np.float32)"""
        arr = self._preprocess(face_img)
        mat = ncnn.Mat.from_pixels(arr, self.PIXEL_TYPE,
                                   self.INPUT_SIZE, self.INPUT_SIZE)
        mat.substract_mean_normalize(self.MEAN, self.NORM)
        ex = self.net.create_extractor()
        ex.input(self.INPUT_BLOB, mat)          # 必须先把图像张量绑到输入层，否则 extract 无数据会崩
        ret, out = ex.extract(self.OUTPUT_BLOB)
        if ret != 0:
            raise RuntimeError("NCNN 提取失败, code=%d" % ret)
        vec = np.array(out, dtype=np.float32).reshape(-1)   # np.array 做拷贝，避免悬空视图
        n = np.linalg.norm(vec)
        if n > 1e-8:
            vec = vec / n
        return vec

    @staticmethod
    def crop_face(full_img, box, scale_out=0.2, target=112):
        """
        从整图按检测框裁剪人脸，并做“外扩(Scale)”（开题报告研究内容②）。
        full_img: PIL.Image(RGB); box: (x1,y1,x2,y2) 像素坐标(float/int)
        返回 PIL.Image(RGB) 裁剪后的人脸（extract 内部会缩放到 target）。
        """
        w, h = full_img.size
        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = x2 - x1, y2 - y1
        x1 -= bw * scale_out
        y1 -= bh * scale_out
        x2 += bw * scale_out
        y2 += bh * scale_out
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(w, int(round(x2)))
        y2 = min(h, int(round(y2)))
        return full_img.crop((x1, y1, x2, y2))


if __name__ == "__main__":
    # 自测：对一张图抽取特征，打印维度与模长，验证模型接线正确
    rec = Recognizer()
    here = os.path.dirname(os.path.abspath(__file__))
    test = os.path.join(here, "..", "test_frame.jpg")
    if os.path.exists(test):
        img = Image.open(test).convert("RGB")
        vec = rec.extract(img)
        print("特征维度:", vec.shape, "模长:", float(np.linalg.norm(vec)))
    else:
        print("未找到测试图，跳过自测")
