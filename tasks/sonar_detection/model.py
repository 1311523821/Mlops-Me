"""
声纳检测模型 —— 包装 LVNet（多帧 3D Swin Transformer 分割网络）。

LVNet 论文: Low-Level Matters: An Efficient Hybrid Architecture for Robust...
输入: (B, D, C, H, W) 多帧灰度图，D=4 帧，C=1 通道，默认 512×512
输出: (B, 1, H, W) 二值分割掩码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_model import BaseModel
from core.registry import register_model


# ---------- 损失函数 ----------

class DiceFocalLoss(nn.Module):
    """Dice + BCE 混合损失，用于极微弱小目标分割"""

    def __init__(self, alpha: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE（带 logits）
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

        # Dice
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        return self.alpha * bce + (1.0 - self.alpha) * dice


# ---------- LVNet 包装 ----------

@register_model("lvnet_tiny")
class LVNetWrapper(BaseModel):
    """
    LVNet 的 MLOps 包装。

    继承 BaseModel，设置 task_type="segmentation" 让 Trainer 使用
    分割专用验证指标（IoU）而非分类准确率。

    构造参数传递到原始 LVNet 类。
    """

    task_type: str = "segmentation"

    def __init__(self, num_frame: int = 4, embed_dim: int = 24,
                 depths: list = None, num_heads: list = None,
                 window_size: tuple = (4, 7, 7)):
        super().__init__()
        # 从同目录导入 LVNet（从 e2e 项目复制而来）
        from .LVNet import LVNet as _LVNet

        depths = depths or [2, 2, 2, 1]
        num_heads = num_heads or [3, 6, 12, 24]

        self.net = _LVNet(
            num_frame=num_frame,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
        )

        self.num_frame = num_frame
        self.input_channels = 1
        self.input_size = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 数据集输出 (B, C, D, H, W)，LVNet 内部 rearrange 到 (B*D, C, H, W)
        return self.net(x)

    def get_example_input(self) -> torch.Tensor:
        """多帧输入: (B=1, C=1, D=4, H=512, W=512)"""
        return torch.randn(1, self.input_channels, self.num_frame,
                           self.input_size, self.input_size)

    def get_loss_fn(self) -> nn.Module:
        return DiceFocalLoss()

    @classmethod
    def from_config(cls, config: dict) -> "LVNetWrapper":
        train_cfg = config.get("training", {})
        return cls(
            num_frame=config.get("_num_frame", 4),
            embed_dim=train_cfg.get("embed_dim", 24),
        )
