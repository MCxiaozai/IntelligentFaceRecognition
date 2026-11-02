import subprocess
import io
import os
import ncnn
import numpy as np
from PIL import Image
from rfb640_utils import prepare_model_image, map_boxes_to_source

def hard_nms(box_scores, iou_threshold=0.3, top_k=-1):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
        
    return box_scores[keep]

def run_face_detection():
    w_img, h_img = 640, 360
    cmd = [
        "v4l2-ctl", "-d", "/dev/video0",
        f"--set-fmt-video=width={w_img},height={h_img},pixelformat=MJPG",
        "--stream-mmap", "--stream-count=1", "--stream-to=-"
    ]
    print("[*] 正在从内核抓取底层画面...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("[-] 抓取失败！")
        return

    # === 关键修复：精准截取二进制流中的 JPEG 图像（过滤 v4l2-ctl 的文本污染） ===
    stdout_data = result.stdout
    start_idx = stdout_data.find(b'\xff\xd8')
    end_idx = stdout_data.find(b'\xff\xd9')
    
    if start_idx != -1 and end_idx != -1:
        jpeg_data = stdout_data[start_idx:end_idx+2]
        img = Image.open(io.BytesIO(jpeg_data)).convert('RGB')
    else:
        print("[-] 未能在输出流中找到有效的 JPEG 图像数据")
        return

    # 初始化 NCNN 模型
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False 
    net.opt.num_threads = 4  
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rfb640")
    net.load_param(os.path.join(model_dir, "RFB-640.param"))
    net.load_model(os.path.join(model_dir, "RFB-640.bin"))

    model_pixels, scale_x, scale_y, pad_left, pad_top = prepare_model_image(img)
    in_pad = ncnn.Mat.from_pixels(
        model_pixels,
        ncnn.Mat.PixelType.PIXEL_RGB,
        640, 480
    )

    mean_vals = [127.0, 127.0, 127.0]
    norm_vals = [1.0 / 127.0, 1.0 / 127.0, 1.0 / 127.0]
    in_pad.substract_mean_normalize(mean_vals, norm_vals)

    ex = net.create_extractor()
    ex.input("in0", in_pad)
    
    ret0, out0 = ex.extract("out1")
    ret1, out1 = ex.extract("out0")
    
    if ret0 == 0 and ret1 == 0:
        normalized_boxes = np.asarray(out0).reshape(-1, 4)
        scores = np.asarray(out1).reshape(-1, 2)
        
        face_scores = scores[:, 1]
        conf_thresh = 0.7
        mask = face_scores > conf_thresh
        
        valid_boxes = normalized_boxes[mask]
        valid_scores = face_scores[mask]
        
        if len(valid_scores) > 0:
            box_scores = np.hstack((valid_boxes, valid_scores[:, np.newaxis]))
            filtered_boxes = hard_nms(box_scores, iou_threshold=0.3)
            source_boxes, valid_indices = map_boxes_to_source(
                filtered_boxes[:, :4], img.size, scale_x, scale_y, pad_left, pad_top
            )
            filtered_boxes = filtered_boxes[valid_indices]
            
            print(f"[+] 成功检测到 {len(filtered_boxes)} 张精准人脸！")
            
            for idx, fb in enumerate(filtered_boxes):
                x1, y1, x2, y2 = source_boxes[idx]
                score = fb[4]
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                print(f"    - 人脸 #{idx+1}: 像素坐标=({x1}, {y1}, {x2}, {y2}), 置信度={score:.4f}")
        else:
            print("[*] 未检测到置信度达标的人脸。")
    else:
        print("[-] 推理执行失败")

if __name__ == "__main__":
    run_face_detection()
