"""RFB-640 人脸检测的图像预处理与坐标映射工具。"""

import numpy as np
from PIL import Image


# 模型训练输入为 640×480；填充值 127 经模型归一化后接近零。
MODEL_WIDTH = 640
MODEL_HEIGHT = 480
PAD_VALUE = (127, 127, 127)


def prepare_model_image(image):
    """等比例缩放并补边，返回模型图像、实际宽高缩放量及左上补边量。"""
    source_width, source_height = image.size
    scale = min(MODEL_WIDTH / source_width, MODEL_HEIGHT / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))

    resized = image.convert("RGB").resize(
        (resized_width, resized_height), Image.BILINEAR
    )
    pad_left = (MODEL_WIDTH - resized_width) // 2
    pad_top = (MODEL_HEIGHT - resized_height) // 2
    scale_x = resized_width / source_width
    scale_y = resized_height / source_height
    model_image = Image.new("RGB", (MODEL_WIDTH, MODEL_HEIGHT), PAD_VALUE)
    model_image.paste(resized, (pad_left, pad_top))
    return np.asarray(model_image), scale_x, scale_y, pad_left, pad_top


def map_boxes_to_source(boxes, source_size, scale_x, scale_y, pad_left, pad_top):
    """把模型输出的归一化边界框映射回原图，并剔除完全落在补边区的框。"""
    source_width, source_height = source_size
    mapped = np.asarray(boxes, dtype=np.float32).copy()
    mapped[:, [0, 2]] = (mapped[:, [0, 2]] * MODEL_WIDTH - pad_left) / scale_x
    mapped[:, [1, 3]] = (mapped[:, [1, 3]] * MODEL_HEIGHT - pad_top) / scale_y

    mapped[:, [0, 2]] = np.clip(mapped[:, [0, 2]], 0, source_width)
    mapped[:, [1, 3]] = np.clip(mapped[:, [1, 3]], 0, source_height)
    valid = (mapped[:, 2] > mapped[:, 0]) & (mapped[:, 3] > mapped[:, 1])
    return mapped[valid], valid
