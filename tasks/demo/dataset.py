"""
Demo 数据集：封装 MNIST，自动下载，支持 --fast 模式取子集。

MNIST 是 28×28 的手写数字灰度图，10 个类别（0-9），
是深度学习领域的 "Hello World"。
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from core.base_dataset import BaseDataset
from core.registry import register_dataset


@register_dataset("mnist")
class MNISTDataset(BaseDataset):
    """
    MNIST 数据集封装。

    自动下载到 data_dir，应用标准化变换。
    --fast 模式时仅返回前 10% 的数据。
    """

    # 类别名映射
    CLASS_NAMES = [str(i) for i in range(10)]

    def __init__(self, data_dir: str, train: bool = True,
                 data_fraction: float = 1.0):
        """
        参数:
            data_dir:       数据存放目录
            train:          是否为训练集
            data_fraction:  使用多少比例的数据（0.0~1.0）
                            --fast 模式时设为 0.1
        """
        self.data_dir = data_dir
        self.train = train
        self.data_fraction = data_fraction

        # MNIST 的均值和标准差（统计好的固定值，直接硬编码）
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        full_dataset = datasets.MNIST(
            root=data_dir, train=train, download=True, transform=transform)

        # --fast 模式：只取前 data_fraction 的数据
        if data_fraction < 1.0:
            num_samples = int(len(full_dataset) * data_fraction)
            full_dataset = Subset(full_dataset, range(num_samples))

        self._dataset = full_dataset
        self._use_subset = data_fraction < 1.0

    @property
    def num_classes(self) -> int:
        return 10

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._dataset[idx]

    def get_preprocess_config(self) -> dict:
        return {
            "mean": [0.1307],
            "std": [0.3081],
            "size": [28, 28],
            "channels": 1,
            "classes": self.CLASS_NAMES,
        }

    @classmethod
    def from_config(cls, config: dict, split: str) -> "MNISTDataset":
        data_dir = config.get("paths", {}).get("data_dir", "./data")
        data_fraction = config.get("_data_fraction", 1.0)

        # split 映射：train/val 都是训练集的不同部分
        # 简单处理：train 用训练集，val 和 test 用测试集
        is_train = (split == "train")

        return cls(data_dir=data_dir, train=is_train,
                   data_fraction=data_fraction)
