"""
模型抽象基类。

所有具体模型必须继承这个类，实现 get_example_input 方法。
get_example_input 在两种情况下会被用到：
1. ONNX 导出时需要 trace 一个示例输入
2. 检查输入尺寸是否匹配
"""

import torch.nn as nn


class BaseModel(nn.Module):
    """
    所有模型的抽象基类。
    直接继承 PyTorch 的 nn.Module，额外要求子类提供 get_example_input。
    """

    def get_example_input(self) -> "torch.Tensor":
        """
        返回一个假的输入张量，形状和真实输入一致。
        供 ONNX 导出、模型检查用。

        子类必须重写这个方法。

        返回:
            torch.Tensor: 示例输入，形状如 (1, 1, 28, 28)
        """
        raise NotImplementedError("子类必须实现 get_example_input 方法")

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
