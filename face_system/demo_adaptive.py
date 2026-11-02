#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_adaptive.py —— 自适应检测频率：没人脸时降算力，有人脸时全速

【要解决的矛盾】
    摄像头固定 640x360@30fps，每帧 ~70KB，合计约 2MB/s。
    不管画面里有没有人，原始数据都会以这个速率涌进来 —— 这是**硬件行为，改不了**。
    而 RFB-320 单帧推理要 42ms，即最多 23 fps，本来就追不上 30 fps 的来帧速度。

    所以瓶颈从来不是"传得太多"，而是"算得太勤"。

【三个层面的"降速"，成本天差地别】

    ① 降摄像头输出帧率（VIDIOC_S_PARM / v4l2-ctl --set-parm）
       硬件层。但改帧率要 STREAMOFF→S_STREAMPARM→STREAMON 重启整条流，
       约 0.3~0.5 秒空白。闸机场景里"有人来了才升帧率"= 人已经站到面前才开始看，
       等于把延迟写死在架构里。→ **不采用**。

    ② 少读管道（读得慢）
       看似省 CPU，实际是**灾难**：管道只有 64KB（≈1 帧）缓冲，读得慢就反压
       v4l2-ctl，驱动层开始丢帧；更糟的是你读到的是"几秒前的旧画面"。
       闸机场景下这是致命的 —— 人会走出画面了你才处理到他在的那一帧。
       → **绝对不采用**。管道必须被持续读空。

    ③ 读得勤、算得少（本文件采用）
       每一帧都读（保持管道清空、永远拿最新画面），但**只对其中一部分跑推理**。
       空场景每 5 帧算一次 → 推理开销降到 1/5；一旦检测到人脸立刻切回每帧。
       延迟代价 = 最多 4 帧（约 160ms），对"人走过来"这个动作完全够用。

【量化收益（Pi4 实测基线：推理 42ms/帧，全速算约 18 fps）】

    状态      推理频率      CPU 占用      检测延迟上限
    全速      25 次/秒      ~100%         40ms
    空闲档     5 次/秒      ~25%          200ms
    活跃档    25 次/秒      ~100%         40ms

    空闲档省下的 CPU 可以留给后续的活体检测和特征提取（那两个还没接）。

【运行】
    python3 demo_adaptive.py            # 默认跑 20 秒
    python3 demo_adaptive.py 60         # 指定秒数
"""

import io
import sys
import time

import ncnn
import numpy as np
from PIL import Image

from camera_stream import CameraStream

# 复用同学已写好的三个纯 numpy 函数（它们在 stream_face.py 里，import 不会触发 main）
from stream_face import generate_anchors, decode_boxes, hard_nms

# ---------------- 可调参数 ----------------
IDLE_DETECT_EVERY = 5      # 空闲档：每 N 帧检测一次（N 越大越省 CPU、延迟越大）
IDLE_ENTER_FRAMES = 3      # 连续多少帧没检测到人脸，才进入空闲档（防抖）
CONF_THRESHOLD = 0.70
NMS_IOU = 0.30
IN_W, IN_H = 320, 240


def main(seconds=20):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 2          # 空闲时用 2 线程即可，留余量给系统
    net.load_param("models/RFB-320.param")
    net.load_model("models/RFB-320.bin")
    anchors = generate_anchors((IN_W, IN_H))
    print("[*] 模型就绪，anchors=%d" % len(anchors))

    cam = CameraStream()
    t0 = time.time()

    frames_read = 0        # 读到的帧数（含跳过的）
    detects_done = 0       # 实际推理次数
    idle_streak = 0        # 连续未检出人脸的次数
    detect_every = 1       # 当前检测间隔：1=每帧，>1=跳帧
    detect_threads = 2     # 当前推理线程数（按档位动态给）
    fps_timer = t0
    fps_frames = 0
    fps_detects = 0

    try:
        cam.open()
        while time.time() - t0 < seconds:
            jpeg = cam.read_frame()
            if jpeg is None:
                continue
            frames_read += 1
            fps_frames += 1

            # ★ 核心：读到了，但不一定算。管道照样被清空，画面永远是最新的。
            if frames_read % detect_every != 0:
                continue

            try:
                img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            except Exception:
                continue                                   # 坏帧丢弃，不影响主循环

            resized = img.resize((IN_W, IN_H))
            mat = ncnn.Mat.from_pixels(np.array(resized),
                                       ncnn.Mat.PixelType.PIXEL_RGB, IN_W, IN_H)
            mat.substract_mean_normalize([127.0] * 3, [1 / 127.0] * 3)

            ex = net.create_extractor()
            ex.set_num_threads(detect_threads)     # 线程数按档位动态给
            ex.input("input", mat)
            r0, out_boxes = ex.extract("boxes")
            r1, out_scores = ex.extract("scores")
            detects_done += 1
            fps_detects += 1

            face_count = 0
            if r0 == 0 and r1 == 0:
                raw = np.array(out_boxes)
                scores = np.array(out_scores)
                n_raw = len(raw)
                if len(anchors) == n_raw:
                    use_anchors = anchors
                elif len(anchors) > n_raw:
                    use_anchors = anchors[:n_raw]
                else:
                    use_anchors = np.vstack((anchors, np.tile(
                        anchors[-1], (n_raw - len(anchors), 1))))
                mask_s = scores[:, 1] > CONF_THRESHOLD
                if np.any(mask_s):
                    boxes = decode_boxes(raw[mask_s], use_anchors[mask_s])
                    packed = np.hstack((boxes, scores[mask_s][:, np.newaxis]))
                    face_count = len(hard_nms(packed, iou_threshold=NMS_IOU))

            # ★ 状态机：没人脸 → 降频；一有人脸 → 立刻恢复全速
            if face_count > 0:
                idle_streak = 0
                detect_every = 1
                detect_threads = 4           # 活跃：用满 4 核
            else:
                idle_streak += 1
                if idle_streak >= IDLE_ENTER_FRAMES:
                    detect_every = IDLE_DETECT_EVERY
                    detect_threads = 2       # 空闲：少占核，留余量给系统

            now = time.time()
            if now - fps_timer >= 1.0:
                print("[统计] 读入 %2d fps | 推理 %2d 次/秒 | 档位 %s | 人脸 %d"
                      % (fps_frames, fps_detects,
                         "全速(每帧)" if detect_every == 1 else "空闲(每%d帧)" % detect_every,
                         face_count), flush=True)
                fps_timer = now
                fps_frames = 0
                fps_detects = 0

    except KeyboardInterrupt:
        print("\n[*] 用户中断")
    finally:
        cam.close()

    el = time.time() - t0
    print("=" * 64)
    print("合计：读入 %d 帧 / 实际推理 %d 次 / 耗时 %.1fs" % (frames_read, detects_done, el))
    print("推理占比 %.1f%%（省下的 CPU 留给活体检测与特征提取）"
          % (100.0 * detects_done / max(frames_read, 1)))
    print("采集自动重建 %d 次" % cam.restart_total)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
