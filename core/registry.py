"""
模型和数据集注册表。

用一个装饰器把类注册到全局字典里，后续通过名字字符串就能取出类。
和 Spring 的 @Component、KubeFlow 的 component registry 思路一致，
这是 MLOps 平台的核心抽象之一。
"""

MODELS = {}       # { "模型名字": 模型类 }
DATASETS = {}     # { "数据集名字": 数据集类 }


def register_model(name: str):
    """
    模型注册装饰器。

    用法:
        @register_model("simple_cnn")
        class SimpleCNN(BaseModel):
            ...
    """
    def wrapper(cls):
        if name in MODELS:
            raise ValueError(f"模型名字 '{name}' 已经被注册了，换一个名字吧")
        MODELS[name] = cls
        return cls
    return wrapper


def register_dataset(name: str):
    """
    数据集注册装饰器。

    用法:
        @register_dataset("mnist")
        class MNISTDataset(BaseDataset):
            ...
    """
    def wrapper(cls):
        if name in DATASETS:
            raise ValueError(f"数据集名字 '{name}' 已经被注册了，换一个名字吧")
        DATASETS[name] = cls
        return cls
    return wrapper


def list_models() -> list:
    """列出所有已注册的模型名字"""
    return list(MODELS.keys())


def list_datasets() -> list:
    """列出所有已注册的数据集名字"""
    return list(DATASETS.keys())
