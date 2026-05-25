"""
Demo 模型：一个极简的 2 层 CNN，用于 MNIST 数据集。

参数量约 10 万，CPU 也能跑，2 分钟内可完成训练。
"""

import torch
import torch.nn as nn
from core.base_model import BaseModel
from core.registry import register_model


@register_model("simple_cnn")
class SimpleCNN(BaseModel):
    """
    极简 CNN，专门为 MNIST 设计。

    网络结构:
        Conv2d(1, 16, 3) → ReLU → MaxPool2d(2)
        Conv2d(16, 32, 3) → ReLU → MaxPool2d(2)
        Flatten → Linear(32*5*5, 128) → ReLU → Dropout(0.3)
        Linear(128, 10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # 两个卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # 池化
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        # 输入 28×28 → 经过两层 MaxPool2d(2) → 7×7 → 32*7*7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_example_input(self) -> torch.Tensor:
        """返回一个 (1, 1, 28, 28) 的假输入，供 ONNX 导出"""
        return torch.randn(1, 1, 28, 28)

    @classmethod
    def from_config(cls, config: dict) -> "SimpleCNN":
        train_cfg = config.get("training", {})
        return cls(
            num_classes=config.get("_num_classes", 10),
            dropout=train_cfg.get("dropout", 0.3),
        )
