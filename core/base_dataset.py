"""
数据集抽象基类。

所有具体数据集必须继承这个类，提供 num_classes 属性和 from_config 工厂方法。
--fast 模式下通过 data_fraction 参数控制数据量。
"""

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    所有数据集的抽象基类。
    继承 PyTorch 的 Dataset，额外要求提供 num_classes 和 from_config。
    """

    @property
    def num_classes(self) -> int:
        """
        返回数据集中的类别总数。
        分类任务必须返回正确的类别数，非分类任务返回 -1。

        子类必须重写这个属性。
        """
        raise NotImplementedError("子类必须实现 num_classes 属性")

    @classmethod
    def from_config(cls, config: dict, split: str) -> "BaseDataset":
        """
        工厂方法：根据配置字典和 split 构建数据集实例。

        参数:
            config: 配置字典（来自 config.yaml）
            split:  数据集划分，'train' / 'val' / 'test'

        返回:
            BaseDataset 实例

        子类必须重写这个方法。
        """
        raise NotImplementedError("子类必须实现 from_config 方法")

    def get_preprocess_config(self) -> dict:
        """
        返回推理时需要的预处理参数。

        这个方法的目的是解耦：export.py 导出 ONNX 时顺带保存这些参数，
        deploy/app.py 启动时读取，不需要知道具体是哪个任务。

        返回:
            dict: 包含以下键
                - mean:     [float, ...]  各通道均值
                - std:      [float, ...]  各通道标准差
                - size:     [int, int]    输入尺寸 (H, W)
                - channels: int           通道数（1=灰度, 3=RGB）
                - classes:  [str, ...]    类别名列表（可选）

        子类应该重写这个方法。
        """
        return {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "size": [224, 224],
            "channels": 3,
            "classes": [str(i) for i in range(self.num_classes)],
        }
