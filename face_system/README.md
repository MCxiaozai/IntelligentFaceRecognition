# 人脸身份识别

## 处理流程

实时识别流程使用 RetinaFace 检测人脸并输出五个关键点；MiniFASNetV2 先判断活体，通过后才进行五点对齐、MobileFaceNet 特征提取和本地人脸库比对。

已有的 RFB-640 检测脚本和模型仍保留。当前身份识别入口使用 RetinaFace；RFB-640 暂作备用检测方案。

## 树莓派运行

确保树莓派已安装 Python 版 NCNN、NumPy、Pillow，并连接摄像头。进入部署后的目录运行：

```bash
cd ~/face_system
python3 enroll.py "张三" 5
python3 recognize_stream.py 30
```

注册命令采集 5 帧并写入 `face_db.npz`。实时识别命令运行 30 秒；省略秒数时持续运行，按 `Ctrl+C` 退出。识别结果写入终端，并将最后一帧保存为 `recognize_out.jpg`。

## 模型文件

- `models/retinaface/face.param` 和 `face.bin`：检测人脸框及五个关键点。
- `models/facenet/mobilefacenet.param` 和 `mobilefacenet.bin`：提取 128 维人脸特征。输入归一化已包含在模型图中，代码不再重复执行。
- `models/rfb640/`：原有 RFB-640 检测模型，保留备用。

活体门槛在 `recognize_stream.py` 和 `enroll.py` 中暂设为 0.7，必须用项目摄像头采集真人、打印照片和屏幕回放样本后校准。活体检测当前按单帧结果拦截，阈值与防伪效果尚未在树莓派上验证。

身份识别阈值在 `recognize_stream.py` 中暂设为 0.4，也需要用已登记人员和非登记人员样本校准。当前代码尚未在树莓派摄像头上完成整链路实测。
