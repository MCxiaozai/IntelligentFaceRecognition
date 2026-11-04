#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face_db.py —— 轻量级本地人脸特征库（纯 NumPy，无 OpenCV）

对应开题报告「研究内容④」：
  - 本地化人脸特征向量序列化存储（.npz）
  - 基于余弦相似度 / 矩阵点乘的高效比对（微秒级）
  - 与本地特征库 NumPy 矩阵点乘比对

存储结构：names(list[str]) + embeddings(ndarray, shape=(N,128), 已 L2 归一化)
因为库内向量均归一化，[库矩阵] @ [查询向量] 即为各人的余弦相似度。
"""
import os
import numpy as np


class FaceDB:
    def __init__(self, path="face_db.npz"):
        self.path = path
        self.names = []          # 与人脸一一对应的姓名列表
        self.embeddings = None   # shape=(N,128) 的 float32 矩阵，行向量已 L2 归一化

    # ---------------- 基本工具 ----------------
    @staticmethod
    def l2_normalize(v):
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(v)
        if n < 1e-8:
            return v
        return v / n

    # ---------------- 注册 ----------------
    def enroll(self, name, embedding):
        """把一个人的一张人脸特征加入库。同名则做向量平均后归一化（每人保留一条代表向量）。"""
        emb = self.l2_normalize(embedding)
        if self.embeddings is None:
            self.embeddings = emb.reshape(1, -1)
            self.names = [name]
            return
        idx = self._find_index(name)
        if idx is not None:
            # 平均已存向量与新向量，再归一化
            merged = self.l2_normalize(self.embeddings[idx] + emb)
            self.embeddings[idx] = merged
        else:
            self.embeddings = np.vstack([self.embeddings, emb.reshape(1, -1)])
            self.names.append(name)

    def _find_index(self, name):
        for i, n in enumerate(self.names):
            if n == name:
                return i
        return None

    # ---------------- 持久化 ----------------
    def save(self, path=None):
        path = path or self.path
        np.savez(path,
                 names=np.array(self.names, dtype=object),
                 embeddings=self.embeddings.astype(np.float32))
        self.path = path

    def load(self, path=None):
        path = path or self.path
        if not os.path.exists(path):
            return False
        data = np.load(path, allow_pickle=True)
        self.names = list(data["names"])
        self.embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        self.path = path
        return True

    # ---------------- 比对 ----------------
    def recognize(self, embedding, threshold=0.4):
        """
        返回 (name, score)。
        score = 余弦相似度 = 归一化查询向量与库矩阵的点乘最大值。
        超过 threshold 取最高分者；否则返回 ('unknown', 最高分)。
        """
        if self.embeddings is None or len(self.names) == 0:
            return "unknown", 0.0
        q = self.l2_normalize(embedding).reshape(1, -1)        # (1,128)
        sims = (self.embeddings @ q.T).reshape(-1)             # (N,) 矩阵点乘
        best = int(np.argmax(sims))
        score = float(sims[best])
        name = self.names[best] if score >= threshold else "unknown"
        return name, score

    def __len__(self):
        return 0 if self.embeddings is None else len(self.names)


if __name__ == "__main__":
    db = FaceDB()
    if db.load():
        print("已加载人脸库：%d 人" % len(db))
        for n in db.names:
            print("  -", n)
    else:
        print("人脸库为空或不存在：%s" % db.path)
