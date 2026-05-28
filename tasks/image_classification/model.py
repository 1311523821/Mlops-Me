"""
图像分类模型定义。

这里注册了几个常用模型作为示例。你也可以在这里添加自己的模型，
只需: @register_model("你的模型名") 然后继承 BaseModel。

已注册的模型:
    - resnet18:  torchvision 的 ResNet-18（适合入门）
    - efficientnet_b0: 轻量高效模型（适合边缘设备部署）
"""

import torch
import torch.nn as nn
from torchvision import models
from core.base_model import BaseModel
from core.registry import register_model


class _TorchvisionWrapper(BaseModel):
    """
    内部辅助类：包装 torchvision 的预训练模型。

    把 torchvision 模型的第一层和最后一层替换掉，
    使之适配自定义的图像尺寸和类别数。
    """

    def __init__(self, model_name: str, num_classes: int,
                 input_channels: int = 3, pretrained: bool = True):
        """
        通过 getattr 动态获取 torchvision 模型，替换首层和末层适配自定义任务。

        参数:
            model_name:     torchvision 模型函数名（如 "resnet18", "efficientnet_b0"）
            num_classes:    输出类别数
            input_channels: 输入通道数（≠3 时替换第一层 Conv2d，保持 kernel/stride/padding 不变）
            pretrained:     是否加载 ImageNet 预训练权重
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_channels = input_channels

        # 根据名字获取 torchvision 模型类
        model_fn = getattr(models, model_name)
        self.backbone = model_fn(weights="DEFAULT" if pretrained else None)

        # 如果输入通道不是 3，替换第一层卷积
        if input_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                input_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None)

        # 替换最后一层全连接
        if hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, 'classifier'):
            if isinstance(self.backbone.classifier, nn.Linear):
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Linear(in_features, num_classes)
            else:
                # EfficientNet 的 classifier 是 Sequential，替换最后一层
                in_features = self.backbone.classifier[-1].in_features
                self.backbone.classifier[-1] = nn.Linear(
                    in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_example_input(self) -> torch.Tensor:
        # 默认 RGB 输入 224×224
        return torch.randn(1, self.input_channels, 224, 224)

    @classmethod
    def from_config(cls, config: dict) -> "BaseModel":
        """从配置构建模型，默认使用 resnet18，类别数和通道数从 config 读取。"""
        return cls(
            model_name="resnet18",
            num_classes=config.get("_num_classes", 10),
            input_channels=3,
            pretrained=True,
        )


# ---------- 注册具体模型 ----------

@register_model("resnet18")
class ResNet18(_TorchvisionWrapper):
    """ResNet-18 图像分类模型"""
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__("resnet18", num_classes, input_channels)

    @classmethod
    def from_config(cls, config: dict) -> "ResNet18":
        return cls(
            num_classes=config.get("_num_classes", 10),
            input_channels=3,
        )


@register_model("efficientnet_b0")
class EfficientNetB0(_TorchvisionWrapper):
    """EfficientNet-B0 轻量分类模型，适合边缘设备"""
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__("efficientnet_b0", num_classes, input_channels)

    @classmethod
    def from_config(cls, config: dict) -> "EfficientNetB0":
        return cls(
            num_classes=config.get("_num_classes", 10),
            input_channels=3,
        )
