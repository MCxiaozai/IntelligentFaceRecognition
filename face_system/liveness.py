#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 NCNN 运行 MiniFASNetV2，输出三类分数并读取真人类别分数。"""
import os
import numpy as np
from PIL import Image
import ncnn

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "liveness")
MODEL_PARAM = os.path.join(MODEL_DIR, "MiniFASNetV2.ncnn.param")
MODEL_BIN = os.path.join(MODEL_DIR, "MiniFASNetV2.ncnn.bin")


class MiniFASNet:
    """根据检测框外扩裁剪人脸，并计算三类活体概率。"""

    INPUT_SIZE = 80
    CROP_SCALE = 2.7
    INPUT_BLOB = "in0"
    OUTPUT_BLOB = "out0"
    REAL_CLASS = 1

    def __init__(self, num_threads=2):
        if not (os.path.isfile(MODEL_PARAM) and os.path.isfile(MODEL_BIN)):
            raise FileNotFoundError("MiniFASNetV2 模型文件缺失: %s" % MODEL_DIR)
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = num_threads
        ret = self.net.load_param(MODEL_PARAM)
        if ret != 0:
            raise RuntimeError("MiniFASNetV2 参数加载失败，错误码: %d" % ret)
        ret = self.net.load_model(MODEL_BIN)
        if ret != 0:
            raise RuntimeError("MiniFASNetV2 权重加载失败，错误码: %d" % ret)

    @staticmethod
    def _crop_face(image, box, scale):
        """按训练时的外扩比例裁剪人脸，并缩放到模型输入尺寸。"""
        width, height = image.size
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        scale = min((height - 1) / box_height, (width - 1) / box_width, scale)
        crop_width = box_width * scale
        crop_height = box_height * scale
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        left = max(0, int(center_x - crop_width / 2.0))
        top = max(0, int(center_y - crop_height / 2.0))
        right = min(width - 1, int(center_x + crop_width / 2.0))
        bottom = min(height - 1, int(center_y + crop_height / 2.0))
        if right <= left or bottom <= top:
            raise ValueError("人脸框无效，无法进行活体裁剪")
        return image.crop((left, top, right + 1, bottom + 1)).resize(
            (MiniFASNet.INPUT_SIZE, MiniFASNet.INPUT_SIZE), Image.BILINEAR)

    def predict(self, image, box):
        """返回三个类别的概率，类别 1 为真人。"""
        face = self._crop_face(image.convert("RGB"), box, self.CROP_SCALE)
        pixels = np.asarray(face, dtype=np.uint8)[:, :, ::-1].copy()
        mat = ncnn.Mat.from_pixels(
            pixels, ncnn.Mat.PixelType.PIXEL_BGR,
            self.INPUT_SIZE, self.INPUT_SIZE)

        ex = self.net.create_extractor()
        ex.input(self.INPUT_BLOB, mat)
        ret, output = ex.extract(self.OUTPUT_BLOB)
        if ret != 0:
            raise RuntimeError("MiniFASNetV2 推理失败，错误码: %d" % ret)

        logits = np.asarray(output, dtype=np.float32).reshape(-1)
        if logits.size != 3 or not np.all(np.isfinite(logits)):
            raise RuntimeError("MiniFASNetV2 输出异常，预期为三个有效分数")
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
