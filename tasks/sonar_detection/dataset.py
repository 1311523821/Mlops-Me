"""
声纳/红外多帧小目标检测数据集。

数据格式（与 e2e 项目兼容）：
    标注文件（txt）：每行一个 DataRecord 文件夹路径
    每个 DataRecord 文件夹内包含连续帧图像和标注数据

输入: 连续 D 帧灰度图，堆叠为 (D, 1, H, W)
输出: 二值分割掩码 (1, H, W)
"""

import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from core.base_dataset import BaseDataset
from core.registry import register_dataset


class SonarFrameDataset(BaseDataset):
    """
    多帧声纳/红外检测数据集。

    兼容 e2e 项目的 txt 标注格式。
    """

    def __init__(self, data_dir: str, split_file: str,
                 num_frame: int = 4, image_size: tuple = (512, 512),
                 data_fraction: float = 1.0):
        """
        参数:
            data_dir:      数据根目录（包含 DataRecord 文件夹）
            split_file:    标注 txt 文件路径（或相对于 data_dir 的路径）
            num_frame:     输入帧数（默认 4，与 LVNet 匹配）
            image_size:    图像缩放尺寸 (H, W)
            data_fraction: 数据使用比例（--fast 模式用）
        """
        self.data_dir = data_dir
        self.num_frame = num_frame
        self.image_size = image_size
        self.data_fraction = data_fraction

        # 解析标注文件
        if not os.path.isabs(split_file):
            split_file = os.path.join(data_dir, split_file)

        self.samples = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(line)

        # --fast 模式：取子集
        if data_fraction < 1.0:
            n = max(1, int(len(self.samples) * data_fraction))
            self.samples = self.samples[:n]

    @property
    def num_classes(self) -> int:
        return 2  # 背景 / 目标（二值分割）

    def __len__(self) -> int:
        # 每个样本序列需要 num_frame 帧，去掉首尾不够的部分
        return max(0, len(self.samples) - self.num_frame + 1)

    def __getitem__(self, idx: int):
        """
        返回 (frames, mask)。

        frames: (D, 1, H, W) 归一化到 [0, 1] 的张量
        mask: (1, H, W) 二值掩码
        """
        H, W = self.image_size
        frames = []

        for offset in range(self.num_frame):
            folder = self.samples[idx + offset]
            folder_path = os.path.join(self.data_dir, folder)

            # 尝试从 DataRecord 文件夹读取图像
            img = self._load_image(folder_path)
            img = cv2.resize(img, (W, H))
            frames.append(img)

        # 堆叠多帧 → (D, H, W)
        frames = np.stack(frames, axis=0)  # (D, H, W)

        # 加载标签（从最后一帧对应的文件夹读取）
        last_folder = self.samples[idx + self.num_frame - 1]
        mask = self._load_mask(os.path.join(self.data_dir, last_folder))
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # 转张量 —— LVNet 期望 (C, D, H, W)，C=通道=1，D=帧数
        frames_t = torch.from_numpy(frames).float().unsqueeze(0)  # (1, D, H, W)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)      # (1, H, W)

        # 归一化到 [0, 1]（简单除以 255）
        frames_t = frames_t / 255.0
        mask_t = (mask_t > 0.5).float()

        return frames_t, mask_t

    def _load_image(self, folder_path: str) -> np.ndarray:
        """从 DataRecord 文件夹加载单帧图像。子类可重写此方法适配不同格式。"""
        # 尝试常见格式
        for fname in ["image.png", "image.jpg", "frame.png", "frame_0000.png"]:
            p = os.path.join(folder_path, fname)
            if os.path.exists(p):
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img

        # 兜底：找第一个 png/jpg
        for f in sorted(os.listdir(folder_path)):
            if f.endswith((".png", ".jpg", ".jpeg")):
                return cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_GRAYSCALE)

        raise FileNotFoundError(f"在 {folder_path} 中找不到图像文件")

    def _load_mask(self, folder_path: str) -> np.ndarray:
        """从 DataRecord 文件夹加载分割掩码。子类可重写。"""
        for fname in ["mask.png", "label.png", "segmentation.png"]:
            p = os.path.join(folder_path, fname)
            if os.path.exists(p):
                return cv2.imread(p, cv2.IMREAD_GRAYSCALE)

        # 兜底：返回全零掩码（可能标签在别处）
        img = self._load_image(folder_path)
        return np.zeros_like(img)

    def get_preprocess_config(self) -> dict:
        return {
            "mean": [0.5],
            "std": [0.5],
            "size": list(self.image_size),
            "channels": self.num_frame,
            "classes": ["background", "target"],
        }

    @classmethod
    def from_config(cls, config: dict, split: str) -> "SonarFrameDataset":
        data_dir = config.get("paths", {}).get("data_dir", "./data")
        data_fraction = config.get("_data_fraction", 1.0)

        # 根据 split 选择对应的 txt 文件
        split_map = {
            "train": config.get("_train_split", "train1.txt"),
            "val": config.get("_val_split", "val_new.txt"),
            "test": config.get("_test_split", "val_new.txt"),
        }
        split_file = split_map.get(split, "train1.txt")

        num_frame = config.get("_num_frame", 4)
        img_size = config.get("_img_size", 512)
        image_size = (img_size, img_size)

        return cls(
            data_dir=data_dir,
            split_file=split_file,
            num_frame=num_frame,
            image_size=image_size,
            data_fraction=data_fraction,
        )


@register_dataset("sonar_e2e")
class RegisteredSonarDataset(SonarFrameDataset):
    """已注册的声纳检测数据集"""
    pass
