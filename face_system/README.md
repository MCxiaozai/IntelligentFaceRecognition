# 人脸身份识别

## 处理流程

实时识别流程使用 RetinaFace 检测人脸并输出五个关键点，再将人脸对齐到 112×112，交给 MobileFaceNet 提取特征，最后与本地特征库计算余弦相似度。

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

首次运行建议先用少量已知人员录入并观察相似度，再按实际数据调整 `recognize_stream.py` 中的识别阈值。当前版本尚未在树莓派摄像头上验证识别准确率和运行帧率。
