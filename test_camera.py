import subprocess
import io
import numpy as np
from PIL import Image

def capture_and_save():
    cmd = [
        "v4l2-ctl", "-d", "/dev/video0",
        "--set-fmt-video=width=640,height=360,pixelformat=MJPG",
        "--stream-mmap", "--stream-count=1", "--stream-to=-"
    ]
    print("[*] 正在抓取摄像头裸流...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print("[-] 抓取失败:", result.stderr.decode())
        return

    # 直接读取字节流并转换为 RGB 张量
    raw_bytes = result.stdout
    img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
    
    # 将张量保存为当前目录下的 JPEG 文件
    save_path = "test_frame.jpg"
    img.save(save_path)
    print(f"[+] 画面已成功捕获并保存为: {save_path}")
    print(f"[+] 张量维度: {np.array(img).shape}")

if __name__ == "__main__":
    capture_and_save()
