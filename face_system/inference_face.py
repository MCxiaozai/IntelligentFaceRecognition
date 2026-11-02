import subprocess
import io
import ncnn
import numpy as np
from PIL import Image

def generate_anchors(image_size=(320, 240)):
    min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    steps = [8, 16, 32, 64]
    
    anchors = []
    for scale, fm_size in enumerate(zip([30, 15, 7, 4], [40, 20, 10, 5])):
        fh, fw = fm_size
        step = steps[scale]
        min_box = min_boxes[scale]
        for h in range(fh):
            for w in range(fw):
                center_x = (w + 0.5) * step
                center_y = (h + 0.5) * step
                for box_size in min_box:
                    s_x = box_size / image_size[0]
                    s_y = box_size / image_size[1]
                    anchors.append([center_x / image_size[0], center_y / image_size[1], s_x, s_y])
                    
    return np.array(anchors, dtype=np.float32)

def decode_boxes(boxes, anchors, center_variance=0.1, size_variance=0.2):
    boxes_centers = boxes[:, :2] * center_variance * anchors[:, 2:] + anchors[:, :2]
    boxes_sizes = np.exp(boxes[:, 2:] * size_variance) * anchors[:, 2:]
    
    boxes_x1y1 = boxes_centers - boxes_sizes / 2.0
    boxes_x2y2 = boxes_centers + boxes_sizes / 2.0
    return np.hstack((boxes_x1y1, boxes_x2y2))

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

    img_matrix = np.array(img)

    # 初始化 NCNN 模型
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False 
    net.opt.num_threads = 4  
    net.load_param("models/RFB-320.param")
    net.load_model("models/RFB-320.bin")

    in_w, in_h = 320, 240
    img_resized = Image.fromarray(img_matrix).resize((in_w, in_h))
    in_pad = ncnn.Mat.from_pixels(
        np.array(img_resized),
        ncnn.Mat.PixelType.PIXEL_RGB,
        in_w, in_h
    )

    mean_vals = [127.0, 127.0, 127.0]
    norm_vals = [1.0 / 127.0, 1.0 / 127.0, 1.0 / 127.0]
    in_pad.substract_mean_normalize(mean_vals, norm_vals)

    ex = net.create_extractor()
    ex.input("input", in_pad)
    
    ret0, out0 = ex.extract("boxes")
    ret1, out1 = ex.extract("scores")
    
    if ret0 == 0 and ret1 == 0:
        raw_boxes = np.array(out0)   
        scores = np.array(out1)      
        
        anchors = generate_anchors((in_w, in_h))
        if len(anchors) > len(raw_boxes):
            anchors = anchors[:len(raw_boxes)]
        elif len(anchors) < len(raw_boxes):
            padding = np.tile(anchors[-1], (len(raw_boxes) - len(anchors), 1))
            anchors = np.vstack((anchors, padding))
        
        face_scores = scores[:, 1]
        conf_thresh = 0.7
        mask = face_scores > conf_thresh
        
        valid_raw_boxes = raw_boxes[mask]
        valid_scores = face_scores[mask]
        valid_anchors = anchors[mask]
        
        if len(valid_scores) > 0:
            decoded_boxes = decode_boxes(valid_raw_boxes, valid_anchors)
            box_scores = np.hstack((decoded_boxes, valid_scores[:, np.newaxis]))
            filtered_boxes = hard_nms(box_scores, iou_threshold=0.3)
            
            print(f"[+] 成功检测到 {len(filtered_boxes)} 张精准人脸！")
            
            for idx, fb in enumerate(filtered_boxes):
                x1, y1, x2, y2, score = fb
                x1 = int(x1 * w_img)
                y1 = int(y1 * h_img)
                x2 = int(x2 * w_img)
                y2 = int(y2 * h_img)
                print(f"    - 人脸 #{idx+1}: 像素坐标=({x1}, {y1}, {x2}, {y2}), 置信度={score:.4f}")
        else:
            print("[*] 未检测到置信度达标的人脸。")
    else:
        print("[-] 推理执行失败")

if __name__ == "__main__":
    run_face_detection()
