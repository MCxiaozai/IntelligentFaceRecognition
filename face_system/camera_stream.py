#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_stream.py —— 树莓派 MJPG 采集封装（带超时重建）

【为什么需要这个文件】
原实现 subprocess + 裸 read(4096) 有两个死穴：
  1. v4l2-ctl 退出 → read 立即返回空 → continue → 100% CPU 空转，永不恢复
  2. v4l2-ctl 活着但不送帧 → read 永久阻塞 → 程序静止，且无法与"没人脸"区分
本文件把「读一帧」封装成有超时、会自愈的操作。

【每条设计对应一个具体的坑】
  1. Popen(bufsize=0) + os.read()
       BufferedReader 会把数据藏在自己的 8KB 缓冲里，select 看不到 fd 就绪，
       造成"明明有数据却判定超时"的假故障。必须绕过缓冲层。
  2. select 超时
       区分「暂时没数据」（等）和「永久断流」（重建）。裸阻塞做不到这件事。
  3. 指数退避重建 0.5→1→2→4→5s
       故障时避免每秒重启几十次把 CPU 打满。
  4. buffer 上限 4MB
       万一标记错乱，防止内存被无限增长吃干（Pi 只有 1.8GB）。
  5. stderr 重定向到文件，不用 PIPE
       stderr=PIPE 而不读，64KB 写满后子进程会阻塞在 write —— 这正是
       "进程活着但不送帧"的一种成因，自己给自己挖坑。
  6. finally 里 close()
       任何退出路径都干掉子进程，否则残留进程占着 /dev/video0，
       下次启动 EBUSY 秒退 —— 这又是另一个死循环的起点。

【用法】
    from camera_stream import CameraStream

    cam = CameraStream()
    try:
        while True:
            jpeg = cam.read_frame()
            if jpeg is None:        # 已自动重建过，跳过本帧即可
                continue
            ...                     # jpeg 是完整可解码的 JPEG bytes
    finally:
        cam.close()

【约定】
    read_frame() 返回 None 只有一种含义：
        「这一帧没拿到，但异常已经处理过了」——调用方只需 continue，
        不需要写任何恢复逻辑，也不需要 sleep。

【自测】
    python3 camera_stream.py 15
    然后另开终端执行  pkill -f v4l2-ctl  观察是否自动恢复
"""

import io
import os
import select
import subprocess
import sys
import time

DEFAULT_DEV = "/dev/video0"
DEFAULT_W, DEFAULT_H = 640, 360
ERRLOG = "/tmp/v4l2_capture.log"
MAX_BUFFER = 4 * 1024 * 1024      # 4MB，约 40 帧的余量


class CameraStream:
    def __init__(self, dev=DEFAULT_DEV, width=DEFAULT_W, height=DEFAULT_H,
                 read_timeout=2.0, base_delay=0.5, max_delay=5.0,
                 reap_on_open=True, verbose=True):
        self.dev = dev
        self.width = width
        self.height = height
        self.read_timeout = read_timeout    # 超过这么久收不到字节 → 判定断流
        self.base_delay = base_delay        # 首次重建等待
        self.max_delay = max_delay          # 退避上限
        self.reap_on_open = reap_on_open    # 打开前清掉占用设备的残留进程
        self.verbose = verbose

        self.proc = None
        self.errlog = None
        self._fd = None
        self._buf = bytearray()

        # 统计信息，便于写进报告
        self.frame_total = 0
        self.restart_total = 0
        self._consecutive_fail = 0

    # ---------------- 内部实现 ----------------

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _cmd(self):
        return [
            "v4l2-ctl", "-d", self.dev,
            "--set-fmt-video=width=%d,height=%d,pixelformat=MJPG" % (self.width, self.height),
            "--stream-mmap",
            "--stream-count=0",        # 需要无限流，别写具体数字
            "--stream-to=-",
        ]

    def _reap_device(self):
        """清掉占着 /dev/video0 的残留采集进程，否则新的 v4l2-ctl 会 EBUSY 秒退。"""
        for cmd in (["fuser", "-k", self.dev], ["pkill", "-f", "v4l2-ctl"]):
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=3)
                time.sleep(0.1)
                return
            except FileNotFoundError:
                continue          # 系统没装 fuser，退到下一个方案
            except Exception:
                continue

    def _spawn(self):
        self.errlog = open(ERRLOG, "ab")
        self.errlog.write(b"\n=== spawn %s | %dx%d ===\n"
                          % (time.strftime("%Y-%m-%d %H:%M:%S").encode(),
                             self.width, self.height))
        self.errlog.flush()
        self.proc = subprocess.Popen(
            self._cmd(),
            stdout=subprocess.PIPE,
            stderr=self.errlog,       # 不读也安全；PIPE 会反压子进程
            bufsize=0,                # 关键：拿到裸 fd，配合 select 使用
        )
        self._fd = self.proc.stdout.fileno()
        self._buf.clear()

    def open(self):
        if self.reap_on_open:
            self._reap_device()
        self._spawn()
        self._log("[采集] v4l2-ctl 已启动 PID=%d，%dx%d MJPG"
                  % (self.proc.pid, self.width, self.height))

    def close(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
        self._fd = None
        if self.errlog is not None:
            try:
                self.errlog.close()
            except Exception:
                pass
            self.errlog = None

    def _extract_frame(self):
        """从缓冲区切出完整一帧；不足一帧返回 None"""
        s = self._buf.find(b"\xff\xd8")
        if s < 0:
            if len(self._buf) > MAX_BUFFER:
                self._buf.clear()
            return None
        if s > 0:
            # 丢掉帧头之前的垃圾 —— 这里是原实现卡死的根源所在：
            # 上一帧切完后可能残留一个孤立的 EOI，不先清掉就会
            # 让"找帧尾"找到帧头之前，条件永远不成立。
            del self._buf[:s]
        e = self._buf.find(b"\xff\xd9")
        if e < 0:
            if len(self._buf) > MAX_BUFFER:
                self._log("[采集] 缓冲区超 4MB 仍无帧尾，丢弃重来")
                self._buf.clear()
            return None
        data = bytes(self._buf[:e + 2])
        del self._buf[:e + 2]
        return data

    def _restart(self, reason):
        self.restart_total += 1
        self._consecutive_fail += 1
        delay = min(self.base_delay * (2 ** (self._consecutive_fail - 1)), self.max_delay)
        self._log("[采集] 异常：%s → %.1fs 后重建（累计第 %d 次）"
                  % (reason, delay, self.restart_total))
        self.close()
        time.sleep(delay)             # 退避：让设备/内核有机会恢复，也避免空转
        try:
            self.open()
        except Exception as e:
            self._log("[采集] 重建失败：%s" % e)

    # ---------------- 对外接口 ----------------

    def read_frame(self):
        """返回一帧完整 JPEG；拿不到（已自动处理）返回 None"""
        if self._fd is None:
            try:
                self.open()
            except Exception as e:
                self._log("[采集] 打开失败：%s" % e)
                time.sleep(self.base_delay)
                return None

        last_data = time.time()
        while True:
            frame = self._extract_frame()
            if frame is not None:
                self.frame_total += 1
                self._consecutive_fail = 0        # 成功即重置退避
                return frame

            idle = time.time() - last_data
            if idle > self.read_timeout:
                self._restart("连续 %.1fs 收不到数据" % idle)
                return None

            try:
                ready, _, _ = select.select([self._fd], [], [], self.read_timeout - idle)
            except (OSError, ValueError) as e:
                self._restart("select 异常 %s" % e)
                return None
            if not ready:
                continue                          # 下一轮命中超时判定

            try:
                chunk = os.read(self._fd, 65536)
            except OSError as e:
                self._restart("read 异常 %s" % e)
                return None
            if not chunk:
                self._restart("EOF，v4l2-ctl 已退出")
                return None
            self._buf.extend(chunk)
            last_data = time.time()               # 收到数据就重置计时

    # ---------------- 自测 ----------------

    def selftest(self, seconds=10):
        from PIL import Image

        print("=" * 64)
        print("自测开始：正常出帧 + 中途杀进程验证自愈")
        print("想测自愈：另开一个终端执行  pkill -f v4l2-ctl")
        print("=" * 64)

        t0 = time.time()
        n = 0
        n_last = 0
        t_last = t0
        try:
            self.open()
            while time.time() - t0 < seconds:
                jpeg = self.read_frame()
                if jpeg is None:
                    continue

                n += 1
                if time.time() - t_last >= 1.0:
                    fps = (n - n_last) / (time.time() - t_last)
                    try:
                        im = Image.open(io.BytesIO(jpeg))
                        info = "%dx%d" % im.size
                    except Exception:
                        info = "解码失败"
                    print("[统计] 累计 %4d 帧 | %5.1f fps | 本帧 %6dB %s | 重启 %d 次"
                          % (n, fps, len(jpeg), info, self.restart_total), flush=True)
                    t_last = time.time()
                    n_last = n
        except KeyboardInterrupt:
            print("\n[自测] 用户中断")
        finally:
            self.close()

        el = time.time() - t0
        print("=" * 64)
        print("合计 %d 帧 / %.1fs = %.1f fps，自动重建 %d 次"
              % (n, el, n / max(el, 1e-6), self.restart_total))


if __name__ == "__main__":
    secs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    CameraStream().selftest(seconds=secs)
