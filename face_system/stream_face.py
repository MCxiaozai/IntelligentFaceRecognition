import io
import time
import ncnn
import numpy as np
from PIL import Image
import subprocess
import os
import select
from rfb640_utils import prepare_model_image, map_boxes_to_source

def hard_nms(box_scores, iou_threshold=0.3, top_k=-1):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    if top_k > 0:
        order = order[:top_k]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        ovr = (w * h) / (areas[i] + areas[order[1:]] - (w * h) + 1e-6)
        order = order[np.where(ovr <= iou_threshold)[0] + 1]
    return box_scores[keep]

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("[*] 正在加载 NCNN 模型...")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False 
    net.opt.num_threads = 4  
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rfb640")
    net.load_param(os.path.join(model_dir, "RFB-640.param"))
    net.load_model(os.path.join(model_dir, "RFB-640.bin"))

    cmd = [
        "v4l2-ctl", "-d", "/dev/video0",
        "--set-fmt-video=width=640,height=360,pixelformat=MJPG",
        "--stream-mmap", "--stream-count=0", "--stream-to=-"
    ]
    
    print("[*] 正在启动底层视频流管道...")
    # bufsize=0: 关掉 Python 的缓冲层。否则 read(4096) 会把数据搬进 BufferedReader
    # 的内部缓冲，此时 select 查的是内核 fd（已空）-> 误判超时 -> 无故重启。
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    
    print("[*] 开始智能休眠控制的实时检测循环 (按 Ctrl+C 退出)...")

    buffer = bytearray()
    frame_count = 0
    no_face_counter = 0  # 连续无脸计数器
    is_sleeping = False  

    try:
        while True:
            t0 = time.time()
            if proc.poll() is not None:
                print("[-] 警告: v4l2-ctl 进程已退出，正在尝试重启...")
                time.sleep(0.5)  # 防止重启失败时变成高频 fork 热循环
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
                buffer.clear()
                continue

            jpeg_data = None
            read_attempts = 0
            
            # 2. 使用 select 轮询管道，设置 2.0 秒超时保护
            # 如果写进程卡死不写数据，select 会直接超时返回空，防止 read() 永远卡死
            # 不能用 0.5s: 实测 v4l2-ctl 启动到首帧要 538ms，0.5 会卡在临界点上无限重启
            rlist, _, _ = select.select([proc.stdout], [], [], 2.0)
            if not rlist:
                print("[-] 警告: 视频流管道读取超时（写进程可能卡死/断流），正在重置管道...")
                proc.terminate()
                proc.wait(timeout=1.0)  # 必须等旧进程真正退出，否则新进程开设备会 EBUSY
                time.sleep(0.5)         # 给驱动释放设备的时间，同时避免热循环
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
                buffer.clear()
                continue

            # 3. 有数据可读时，安全读取并切片
            while read_attempts < 10:  
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                buffer.extend(chunk)
                read_attempts += 1
                
                # 缓冲区溢出熔断保护
                if len(buffer) > 512 * 1024:
                    print("[-] 警告: 管道缓冲区过大，正在清空重置...")
                    buffer.clear()
                    break

                start_idx = buffer.find(b'\xff\xd8')
                if start_idx != -1:
                    if start_idx > 0:
                        del buffer[:start_idx]
                        start_idx = 0
                    
                    end_idx = buffer.find(b'\xff\xd9')
                    if end_idx != -1 and end_idx > start_idx:
                        jpeg_data = bytes(buffer[:end_idx+2])
                        del buffer[:end_idx+2]
                        break
            
            if not jpeg_data:
                continue
            t1 = time.time()
            img = Image.open(io.BytesIO(jpeg_data)).convert('RGB')
            t2 = time.time()

            model_pixels, scale_x, scale_y, pad_left, pad_top = prepare_model_image(img)
            in_pad = ncnn.Mat.from_pixels(
                model_pixels,
                ncnn.Mat.PixelType.PIXEL_RGB,
                640, 480
            )
            in_pad.substract_mean_normalize([127.0]*3, [1/127.0]*3)

            ex = net.create_extractor()
            ex.input("in0", in_pad)
            ret0, out0 = ex.extract("out1")
            ret1, out1 = ex.extract("out0")
            
            t3 = time.time()

            face_count = 0
            if ret0 == 0 and ret1 == 0:
                normalized_boxes = np.asarray(out0).reshape(-1, 4)
                scores = np.asarray(out1).reshape(-1, 2)
                
                face_scores = scores[:, 1]
                mask = face_scores > 0.75
                if np.any(mask):
                    box_scores = np.hstack((normalized_boxes[mask], face_scores[mask][:, np.newaxis]))
                    filtered = hard_nms(box_scores, iou_threshold=0.3)
                    _, valid_boxes = map_boxes_to_source(
                        filtered[:, :4], img.size, scale_x, scale_y, pad_left, pad_top
                    )
                    face_count = int(np.count_nonzero(valid_boxes))

            # === 智能休眠逻辑判定 ===
            if face_count > 0:
                if is_sleeping:
                    print("[*] 检测到人脸，已从休眠中唤醒，恢复全速运行！")
                    is_sleeping = False
                no_face_counter = 0
                status_str = f"检测到: {face_count}个"
            else:
                no_face_counter += 1
                # 如果连续 30 帧（约 1.5 秒）没看到人脸，进入低功耗休眠
                if no_face_counter > 30 and not is_sleeping:
                    is_sleeping = True
                    print("[*] 长期未检测到人脸，进入低功耗休眠模式...")
                status_str = "休眠中 (等待人脸...)" if is_sleeping else "无目标"

            grab_ms = (t1 - t0) * 1000
            infer_ms = (t3 - t2) * 1000
            
            print(f"[帧 {frame_count}] 抓取: {grab_ms:.1f}ms | 推理: {infer_ms:.1f}ms | 状态: {status_str}")

            frame_count += 1

            # 如果进入休眠状态，每帧主动休眠 0.1 秒释放 CPU，防止满载发热
            if is_sleeping:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[*] 正在安全关闭视频流并退出...")
        proc.terminate()
        proc.wait(timeout=1.0)  # 等子进程真正退出再走，否则残留的 v4l2-ctl 会占住摄像头

if __name__ == "__main__":
    main()
