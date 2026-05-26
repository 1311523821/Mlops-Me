---
name: add-mlops-project
description: |
  把外部 GitHub 项目或本地深度学习代码集成到 MLOps 工作流。
  当用户说"把这个项目加入 mlops"、"集成到工作流"、"添加为任务"、
  或提到把某个模型/数据集包装进 MLOps 框架时触发。
  支持分类和分割两种任务类型。
---

# 把外部项目集成到 MLOps 工作流

## 核心原则

**不要硬编码参数。** 所有配置值都从原代码分析出来，MLOps 只做包装不做假设。
参数名、参数值、输入形状、预处理参数 — 一切以原项目为准。

## 流程概览

```
分析项目 → 创建 tasks/<name>/ → 写 model.py → 写 dataset.py
→ 复制依赖 → 创建 tasks/<name>/config.yaml → 验证
```

## 步骤 1：分析项目

先拉取项目，回答以下问题：

```bash
git clone --depth 1 <repo-url> /tmp/project
```

| 问题 | 去哪里找 | 怎么确定 |
|------|----------|----------|
| 模型类名和构造函数参数？ | `class XXX(nn.Module)` | 看 `__init__` 的签名，每个参数的值从默认值或 argparse 中取 |
| 输入形状？ | `forward()` 第一行 + 训练循环 | 打印第一个 batch 的 shape，或看 Dataset 的 `__getitem__` |
| 输出形状？ | `forward()` 最后一行 | `print(output.shape)` |
| 任务类型？ | 损失函数 | CrossEntropyLoss → classification；BCE/Dice/IoU → segmentation |
| 数据格式？ | Dataset 类或训练脚本 | txt/json/文件夹结构 → 决定 dataset.py 怎么读 |
| 本地依赖？ | import 语句 | 非 pip 安装的 `.py` 文件都要复制 |
| 预处理参数？ | Dataset 的 transform | mean、std、resize 尺寸，全部从原代码提取 |

**关键：不要猜测，全部从原代码中复制。**

## 步骤 2：创建任务目录

```bash
mkdir -p tasks/<task_name>
```

最终结构：

```
tasks/<task_name>/
├── __init__.py       # 声明默认模型和数据集
├── model.py          # 包装外部模型
├── dataset.py        # 包装数据加载
├── config.yaml       # 任务级配置（从原代码参数生成）
└── <外部依赖文件>.py  # 原项目的本地模块
```

## 步骤 3：写 __init__.py

```python
DEFAULT_DATASET = "your_dataset_name"
DEFAULT_MODEL = "your_model_name"
```

## 步骤 4：写 model.py

核心任务：把外部模型包装成 `BaseModel`，让 Trainer 能自动获取 loss 和任务类型。

**第一步：从原代码提取构造函数参数**

打开原项目的模型文件，找到 `__init__` 签名，把每个参数抄下来作为 wrapper 的 `__init__` 参数，默认值保持一致。

**第二步：确定 task_type**

| 原代码用的 loss | task_type | get_loss_fn 返回 |
|----------------|-----------|-----------------|
| CrossEntropyLoss | `"classification"` | 不需要重写（默认） |
| BCEWithLogitsLoss | `"segmentation"` | DiceLoss 或自定义 |
| DiceLoss / FocalLoss | `"segmentation"` | 原项目的 loss 类 |

**第三步：确定输入形状**

从原项目的 Dataset `__getitem__` 或训练循环中找到第一个 batch 的 shape，写入 `get_example_input()`。

**模板：**

```python
import torch.nn as nn
from core.base_model import BaseModel
from core.registry import register_model

@register_model("从原项目的模型名推断")
class YourModelWrapper(BaseModel):
    task_type = "从原项目的 loss 推断"

    def __init__(self, <从原代码 __init__ 签名复制的参数>):
        super().__init__()
        # 导入外部模型（注意处理本地导入路径问题，见步骤 6）
        from .external_model import ExternalModel
        self.net = ExternalModel(<参数透传>)

    def forward(self, x):
        return self.net(x)

    def get_example_input(self):
        import torch
        # 形状从原项目 Dataset.__getitem__ 或训练循环中获取
        return torch.randn(<从原代码确定的输入形状>)

    def get_loss_fn(self):
        # 仅分割任务需要重写；分类任务用默认 CrossEntropyLoss
        return <从原代码复制的 loss 类>()

    @classmethod
    def from_config(cls, config):
        # 从 config 中读取参数。键名从原代码 argparse 参数名推断。
        # 分类任务至少传入 num_classes；分割任务传入原模型的所有构造参数。
        return cls(<参数映射>)
```

**容易踩的坑：**
- 输入形状不匹配：模型要 4D 但数据集给了 5D，在 `forward()` 里做 reshape
- 忘记设 `task_type`：默认 classification，分割任务会算错指标
- 忘记重写 `get_loss_fn()`：默认 CrossEntropyLoss 不能用于分割
- 外部模型的本地 import 路径：见步骤 6

## 步骤 5：写 dataset.py

核心任务：把原项目的数据加载逻辑封装成 `BaseDataset`。

**第一步：分析原项目的数据格式**

仔细看原项目的 Dataset 类或训练脚本中的数据读取代码，回答：
- 数据文件类型？（图片/txt/npy/...）
- 目录结构？（文件夹分类 / txt 标注路径 / json 标注 / ...）
- 标签格式？（整数类别 / 二值掩码 / bbox / ...）
- 预处理参数？（mean、std、resize 尺寸 — 全部从原代码提取）

**第二步：把数据加载逻辑搬进 `__getitem__` 和 `__init__`**

**第三步：`get_preprocess_config()` 返回从原代码提取的预处理参数**

**模板：**

```python
from core.base_dataset import BaseDataset
from core.registry import register_dataset

@register_dataset("从原项目推断的名字")
class YourDataset(BaseDataset):
    # 类别名从原项目获取（如果是分类任务）
    CLASS_NAMES = <从原代码提取的类别列表>

    def __init__(self, data_dir, train=True, data_fraction=1.0,
                 <从原代码提取的其他参数，如 image_size、标注文件路径等>):
        self.data_dir = data_dir
        self.train = train
        # 搬运原项目的加载逻辑
        self.samples = <原项目的数据加载代码>

        if data_fraction < 1.0:
            n = max(1, int(len(self.samples) * data_fraction))
            self.samples = self.samples[:n]

    @property
    def num_classes(self):
        return <从原代码确定的类别数>

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 搬运原项目的 __getitem__ 逻辑
        # 返回 (input_tensor, target_tensor)
        ...

    def get_preprocess_config(self):
        """返回推理预处理参数，所有值从原代码提取"""
        return {
            "mean": <从原代码 transform 提取>,
            "std": <从原代码 transform 提取>,
            "size": <从原代码 resize 提取>,
            "channels": <从原代码输入通道数提取>,
            "classes": self.CLASS_NAMES,
        }

    @classmethod
    def from_config(cls, config, split):
        return cls(
            data_dir=config["paths"]["data_dir"],
            train=(split == "train"),
            data_fraction=config.get("_data_fraction", 1.0),
            <透传其他参数>,
        )
```

## 步骤 6：复制依赖文件

检查原项目的 import 语句，把非 pip 包的本地 `.py` 文件复制到 `tasks/<task_name>/` 下。

**如果外部模块内部有绝对导入**（如 `from LVNet import xxx`），在 `model.py` 的 `__init__` 开头加入：

```python
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from ExternalModule import ExternalModel
```

## 步骤 7：创建任务级配置文件

**不要改全局 `config.yaml`**。创建 `tasks/<task_name>/config.yaml`：

1. **fast/full 段**：必填 `model_name` 和 `dataset_name`（与注册名一致）。其他如 `epochs`、`batch_size`、`data_fraction` 按需覆盖全局默认值。

2. **模型参数段**：如果原项目通过 argparse 传入大量模型构造参数（消融实验），用 `model_params` 段存放，字段名和原 argparse 参数名保持一致。

3. **分割参数段**：如果是分割任务且有多帧/标注文件等参数，用 `segmentation` 段存放，字段名从原代码推断。

**原则：config 里的字段名反映原项目的参数名，不做翻译。原代码叫什么，config 里就叫什么。**

```yaml
# tasks/your_task/config.yaml

fast:
  model_name: "和 @register_model 一致"
  dataset_name: "和 @register_dataset 一致"
  epochs: <快速验证 epoch 数，通常 3>
  data_fraction: 0.1

full:
  model_name: "和 @register_model 一致"
  dataset_name: "和 @register_dataset 一致"
  epochs: <原项目默认 epoch 数>

# 以下是按需添加的段，字段名从原代码推断：

# 模型构造参数（如果有的话）
model_params:
  <从原 argparse 复制的参数名>: <从原 argparse 复制的默认值>
  ...

# 数据集参数（如果有的话）
dataset_params:
  <从原代码复制的参数名>: <值>
  ...

# 损失函数参数（如果自定义 loss 有参数）
loss_params:
  <参数名>: <值>
  ...
```

配置合并规则：`tasks/<name>/config.yaml` 覆盖 `config.yaml` 同名字段。`train.py`/`evaluate.py`/`export.py` 自动合并。

## 步骤 8：验证

```bash
# 1. 测试导入链和前向传播
python -c "
import importlib
importlib.import_module('tasks.your_task.model')
importlib.import_module('tasks.your_task.dataset')
from core.registry import MODELS, DATASETS
m = MODELS['your_model_name']()
print('task_type:', m.task_type)
print('loss:', type(m.get_loss_fn()).__name__)
import torch
x = m.get_example_input()
with torch.no_grad():
    y = m(x)
print('Input:', list(x.shape), '→ Output:', list(y.shape))
"

# 2. 有数据的话快速训练
python train.py --task your_task --fast
```

## 真实案例：e2e → sonar_detection

展示如何从原代码分析出所有参数：

**分析过程：**
1. 原项目 `LVNet.py` → 模型类 `LVNet(nn.Module)`，`__init__` 有 16 个参数（num_frame、embed_dim、encoder_type...）
2. 训练脚本用 DiceLoss + BCE → task_type = `"segmentation"`
3. Dataset 返回 `(D, H, W)` 帧堆叠 + 二值掩码 → 输入 5D
4. 标注文件 `train1.txt`/`val_new.txt`，每行列出一个 DataRecord 文件夹路径
5. 依赖：`LVNet.py`、`muon.py`、`sonar_utils.py`

**实现要点：**
- `task_type = "segmentation"`，`get_loss_fn()` 复制原项目的 DiceFocalLoss
- Dataset 的 `from_config` 接收 `split`，映射到对应的标注 txt 文件
- 预处理参数全部从原代码的 transform 提取
- config.yaml 的 `model_params` 段直接从原 argparse 参数名搬运：`encoder_type`、`mlp_type`、`block_type` 等
- LVNet.py 内部有 `from LVNet import ...` 绝对导入，在 model.py 里加了 `sys.path` 修复

**结果：** `python train.py --task sonar_detection --fast` 可直接训练。改 `tasks/sonar_detection/config.yaml` 里 `model_params` 的一个值即可做消融实验，MLflow 自动对比。
