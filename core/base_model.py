"""
模型抽象基类。

支持分类、分割等多种任务类型，通过 task_type 和 get_loss_fn 让 Trainer 不硬编码。
"""

import torch.nn as nn


class BaseModel(nn.Module):
    """
    所有模型的抽象基类。
    继承 PyTorch nn.Module，额外提供 ONNX 导出支持和任务类型声明。
    """

    # 任务类型：子类覆盖此属性声明任务类型
    # "classification" → 分类（CrossEntropyLoss + accuracy）
    # "segmentation"   → 分割（自定义 loss + IoU/PD/FA）
    task_type: str = "classification"

    def get_example_input(self) -> "torch.Tensor":
        """
        返回一个假的输入张量，形状和真实输入一致。
        供 ONNX 导出、模型检查用。

        子类必须重写这个方法。
        """
        raise NotImplementedError("子类必须实现 get_example_input 方法")

    def get_loss_fn(self) -> nn.Module:
        """
        返回该任务使用的损失函数。
        默认返回 CrossEntropyLoss（分类任务）。
        分割任务应重写此方法返回 DiceLoss 等。

        Trainer 调用此方法获取 loss，因此 Trainer 不需要知道具体任务。
        """
        return nn.CrossEntropyLoss()

    @classmethod
    def from_config(cls, config: dict) -> "BaseModel":
        """
        工厂方法：根据配置字典构建模型实例。

        子类可以重写这个方法，从 config 中读取超参数（如 num_classes、dropout 等）
        然后构建自定义模型。默认实现直接无参实例化。

        参数:
            config: 配置字典（来自 config.yaml）

        返回:
            BaseModel 实例
        """
        return cls()
