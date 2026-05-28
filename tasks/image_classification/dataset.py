"""
图像分类数据集定义。

支持两种数据来源:
    - image_folder: 标准 ImageFolder 文件夹结构（data/class_0/, data/class_1/...）
    - CIFAR-10:     torchvision 自动下载

后续可以在这里添加更多数据集格式（CSV 标注、JSON 标注等）。
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from core.base_dataset import BaseDataset
from core.registry import register_dataset


class ImageFolderDataset(BaseDataset):
    """
    通用 ImageFolder 数据集。

    数据目录结构要求:
        data_dir/
        ├── train/
        │   ├── cat/
        │   ├── dog/
        │   └── ...
        └── val/
            ├── cat/
            ├── dog/
            └── ...
    """

    def __init__(self, root_dir: str, data_fraction: float = 1.0,
                 image_size: int = 224):
        """
        参数:
            root_dir:      数据根目录，子目录名为类别（ImageFolder 格式）
            data_fraction: 使用数据的比例（--fast 模式设为 < 1.0）
            image_size:    图像缩放尺寸，默认 224×224
        """
        self.root_dir = root_dir
        self.data_fraction = data_fraction
        self.image_size = image_size

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])

        full_dataset = datasets.ImageFolder(root=root_dir, transform=transform)

        if data_fraction < 1.0:
            num_samples = int(len(full_dataset) * data_fraction)
            full_dataset = Subset(full_dataset, range(num_samples))

        self._dataset = full_dataset
        self._classes = full_dataset.classes if hasattr(full_dataset, 'classes') else []

    @property
    def num_classes(self) -> int:
        if hasattr(self._dataset, 'classes'):
            return len(self._dataset.classes)
        # Subset 情况下需要特殊处理
        return len(self._classes)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._dataset[idx]


class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 数据集封装"""

    def __init__(self, data_dir: str, train: bool = True,
                 data_fraction: float = 1.0, image_size: int = 32):
        self.data_dir = data_dir
        self.train = train
        self.data_fraction = data_fraction
        self.image_size = image_size

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]),
        ])

        full_dataset = datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform)

        if data_fraction < 1.0:
            num_samples = int(len(full_dataset) * data_fraction)
            full_dataset = Subset(full_dataset, range(num_samples))

        self._dataset = full_dataset

    @property
    def num_classes(self) -> int:
        return 10

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._dataset[idx]


# ---------- 注册数据集 ----------

@register_dataset("image_folder")
class RegisteredImageFolder(ImageFolderDataset):
    """已注册的 ImageFolder 数据集"""
    @classmethod
    def from_config(cls, config: dict, split: str) -> "RegisteredImageFolder":
        data_dir = config.get("paths", {}).get("data_dir", "./data")
        data_fraction = config.get("_data_fraction", 1.0)
        image_size = config.get("dataset", {}).get("image_size", 224)
        return cls(root_dir=data_dir, data_fraction=data_fraction,
                   image_size=image_size)

    def get_preprocess_config(self) -> dict:
        return {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "size": [self.image_size, self.image_size],
            "channels": 3,
            "classes": getattr(self, '_classes', [str(i) for i in range(self.num_classes)]),
        }


@register_dataset("cifar10")
class RegisteredCIFAR10(CIFAR10Dataset):
    """已注册的 CIFAR-10 数据集"""
    CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]

    @classmethod
    def from_config(cls, config: dict, split: str) -> "RegisteredCIFAR10":
        data_dir = config.get("paths", {}).get("data_dir", "./data")
        data_fraction = config.get("_data_fraction", 1.0)
        image_size = config.get("dataset", {}).get("image_size", 32)
        is_train = (split == "train")
        return cls(data_dir=data_dir, train=is_train,
                   data_fraction=data_fraction, image_size=image_size)

    def get_preprocess_config(self) -> dict:
        return {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616],
            "size": [self.image_size, self.image_size],
            "channels": 3,
            "classes": self.CIFAR10_CLASSES,
        }
