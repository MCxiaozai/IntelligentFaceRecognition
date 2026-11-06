# MiniFASNetV2 活体模型

## 文件

- `MiniFASNetV2.ncnn.param`、`MiniFASNetV2.ncnn.bin`：树莓派 NCNN 推理文件。
- `MiniFASNetV2.onnx`：用于复现 NCNN 转换的 ONNX 源模型。

## 模型接口

- 输入：80×80、BGR 三通道像素；按检测框中心外扩 2.7 倍后裁剪，直接缩放，不做均值或幅度归一化。
- 输出：三类 logits，代码使用 softmax 转成概率；第 1 类为真人，另两类为攻击类别。
- 检测框外扩比例、通道顺序和类别映射沿用 MiniFASNetV2 的 ONNX 推理实现。

## 来源与转换

- 原始模型项目：[MiniVision Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- ONNX 导出与推理参考：[yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing)
- ONNX 下载地址：[MiniFASNetV2.onnx](https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx)
- ONNX 文件 SHA-256：`b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907`
- NCNN 文件由项目 PNNX 转换生成，输入形状为 `[1,3,80,80]`。

这是单目 RGB 摄像头的被动活体模型。阈值需用本项目摄像头和打印、屏幕回放攻击样本校准，不能把通用模型分数当作已验收的防伪指标。
