"""代码生成器 — 根据分析报告生成 MLOps tasks/<name>/ 全部文件。"""

import os
import re
import shutil

_SAFE_NAME_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]+$')
_ALLOWED_LOSS_TYPES = {"CrossEntropyLoss", "BCEWithLogitsLoss", "DiceLoss",
                       "FocalLoss", "MSELoss", "L1Loss", "CTCLoss", "KLDivLoss"}


def _validate_task_name(name: str) -> None:
    if not _SAFE_NAME_RE.match(name):
        raise ValueError(
            f"非法的任务名: {name!r}，只允许 a-z A-Z 0-9 _ -")
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError(f"任务名不能包含路径分隔符: {name!r}")


def generate_task(task_name: str, report: dict, output_base: str) -> list[str]:
    """
    根据分析报告生成全部任务文件。

    参数:
        task_name: 任务目录名
        report: 分析报告 JSON
        output_base: MLOps 项目根目录

    返回:
        生成的文件路径列表
    """
    _validate_task_name(task_name)
    task_dir = os.path.join(output_base, "tasks", task_name)
    os.makedirs(task_dir, exist_ok=True)

    files = []

    # __init__.py
    init_path = os.path.join(task_dir, "__init__.py")
    with open(init_path, "w", encoding="utf-8") as f:
        f.write(_render_init(task_name))
    files.append(init_path)

    # model.py
    model_path = os.path.join(task_dir, "model.py")
    with open(model_path, "w", encoding="utf-8") as f:
        f.write(_render_model(task_name, report))
    files.append(model_path)

    # dataset.py
    dataset_path = os.path.join(task_dir, "dataset.py")
    with open(dataset_path, "w", encoding="utf-8") as f:
        f.write(_render_dataset(task_name, report))
    files.append(dataset_path)

    # config.yaml
    config_path = os.path.join(task_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(_render_config(task_name, report))
    files.append(config_path)

    return files


def copy_dependencies(report: dict, source_repo: str, task_name: str, output_base: str) -> list[str]:
    """将外部仓库的本地依赖文件复制到任务目录。"""
    _validate_task_name(task_name)
    task_dir = os.path.join(output_base, "tasks", task_name)
    copied = []
    safe_repo = os.path.realpath(source_repo)
    for rel_path in report.get("dependencies", {}).get("local_files", []):
        # 防止路径穿越
        if ".." in rel_path or rel_path.startswith("/") or rel_path.startswith("\\"):
            continue
        src = os.path.join(source_repo, rel_path)
        safe_src = os.path.realpath(src)
        if not safe_src.startswith(safe_repo + os.sep):
            continue
        dst = os.path.join(task_dir, os.path.basename(rel_path))
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def _render_init(task_name: str) -> str:
    return f'''"""任务 {task_name} — 由 Paper2MLOps 自动生成。"""

DEFAULT_DATASET = "{task_name}"
DEFAULT_MODEL = "{task_name}"
'''


def _render_model(task_name: str, report: dict) -> str:
    model = report.get("model", {})
    class_name = model.get("class_name", "UnknownModel")
    if not _SAFE_NAME_RE.match(class_name):
        raise ValueError(f"非法的类名: {class_name!r}，只允许 a-z A-Z 0-9 _ -")
    loss = report.get("loss", {})
    loss_type = loss.get("type", "CrossEntropyLoss")
    if loss_type not in _ALLOWED_LOSS_TYPES:
        loss_type = "CrossEntropyLoss"
    task_type = report.get("task_type", "classification")
    input_shape = model.get("input_shape", [1, 3, 224, 224])
    init_params = model.get("init_params", {})
    source_file = model.get("source_file", "")

    params_str = ", ".join(f"{k}={repr(v)}" for k, v in init_params.items())

    # 判断是否需要自定义 loss
    needs_custom_loss = task_type == "segmentation" or loss_type not in ("CrossEntropyLoss",)

    custom_loss_block = ""
    if needs_custom_loss:
        custom_loss_block = f"""
    def get_loss_fn(self):
        # 从原项目复制自定义 loss 实现
        # TODO: 如有本地 loss 文件，在此导入
        import torch.nn as nn
        return nn.{loss_type}()"""

    return f'''"""任务 {task_name} 的模型包装器 — 由 Paper2MLOps 自动生成。"""

import torch.nn as nn
from core.base_model import BaseModel
from core.registry import register_model


@register_model("{task_name}")
class {class_name}Wrapper(BaseModel):
    """包装自 {source_file} 的 {class_name} 模型。"""
    task_type = "{task_type}"

    def __init__(self, {params_str}):
        super().__init__()
        # 如原模型在本地文件中，取消下行注释并调整导入路径
        # import sys, os
        # _here = os.path.dirname(os.path.abspath(__file__))
        # if _here not in sys.path:
        #     sys.path.insert(0, _here)
        # from {os.path.splitext(os.path.basename(source_file))[0] if source_file else "model_file"} import {class_name}
        # self.net = {class_name}({", ".join(f"{k}={k}" for k in init_params)})
        self.net = nn.Identity()  # 占位，替换为实际模型

    def forward(self, x):
        return self.net(x)

    def get_example_input(self):
        import torch
        return torch.randn({input_shape}){custom_loss_block}

    @classmethod
    def from_config(cls, config):
        model_params = config.get("model_params", {{}})
        return cls(**model_params)
'''


def _render_dataset(task_name: str, report: dict) -> str:
    dataset = report.get("dataset", {})
    num_classes = dataset.get("num_classes", 10)
    class_names = dataset.get("class_names", [str(i) for i in range(num_classes)])
    preprocess = report.get("preprocess", {})

    return f'''"""任务 {task_name} 的数据集 — 由 Paper2MLOps 自动生成。"""

import os
from core.base_dataset import BaseDataset
from core.registry import register_dataset


@register_dataset("{task_name}")
class {task_name.capitalize()}Dataset(BaseDataset):
    """数据集包装器。"""

    CLASS_NAMES = {class_names}

    def __init__(self, data_dir, train=True, data_fraction=1.0):
        self.data_dir = data_dir
        self.train = train
        # TODO: 根据分析报告补充数据加载逻辑
        # 数据格式: {dataset.get('format', '未知')}
        self.samples = []

        if data_fraction < 1.0:
            n = max(1, int(len(self.samples) * data_fraction))
            self.samples = self.samples[:n]

    @property
    def num_classes(self):
        return {num_classes}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # TODO: 实现具体的数据加载逻辑
        raise NotImplementedError("需要根据实际数据格式补充 __getitem__ 实现")

    def get_preprocess_config(self):
        return {{
            "mean": {preprocess.get('mean', [0.5, 0.5, 0.5])},
            "std": {preprocess.get('std', [0.5, 0.5, 0.5])},
            "size": {preprocess.get('size', [224, 224])},
            "channels": {preprocess.get('channels', 3)},
            "classes": self.CLASS_NAMES,
        }}

    @classmethod
    def from_config(cls, config, split):
        return cls(
            data_dir=config["paths"]["data_dir"],
            train=(split == "train"),
            data_fraction=config.get("_data_fraction", 1.0),
        )
'''


def _render_config(task_name: str, report: dict) -> str:
    model = report.get("model", {})
    init_params = model.get("init_params", {})

    params_yaml = ""
    if init_params:
        params_yaml = "\nmodel_params:\n"
        for k, v in init_params.items():
            if isinstance(v, str):
                params_yaml += f'  {k}: "{v}"\n'
            else:
                params_yaml += f"  {k}: {v}\n"

    return f'''# 任务 {task_name} 配置 — 由 Paper2MLOps 自动生成
# 任务类型: {report.get("task_type", "classification")}

fast:
  model_name: "{task_name}"
  dataset_name: "{task_name}"
  epochs: 3
  data_fraction: 0.1

full:
  model_name: "{task_name}"
  dataset_name: "{task_name}"
  epochs: 50
{params_yaml}
'''
