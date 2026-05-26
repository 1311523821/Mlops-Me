---
name: add-mlops-project
description: |
  把外部 GitHub 项目或本地深度学习代码集成到 MLOps 工作流。
  当用户说"把这个项目加入 mlops"、"集成到工作流"、"添加为任务"、
  或提到把某个模型/数据集包装进 MLOps 框架时触发。
  支持分类和分割两种任务类型。
---

# 把外部项目集成到 MLOps 工作流

## 流程概览

```
分析项目 → 创建 tasks/<name>/ → 写 model.py → 写 dataset.py
→ 复制依赖 → 更新 config.yaml → 验证
```

## 步骤 1：分析项目

先拉取项目，理解 3 个关键点：

```bash
git clone --depth 1 <repo-url> /tmp/project
```

**必须弄清楚：**

| 问题 | 去哪里找 |
|------|----------|
| 模型类名是什么？构造函数参数？ | 找 `class XXX(nn.Module)` |
| 输入输出形状？ | 看 `forward()` 和训练循环 |
| 任务类型？ | 分类=CrossEntropyLoss/accuracy，分割=BCE/Dice/IoU |
| 数据怎么加载？ | 找 Dataset 类或训练脚本中的数据读取逻辑 |
| 有哪些依赖文件？ | 看 import 语句，识别本地模块 |

**关键判断：task_type**

```python
# 分类 → task_type = "classification"
# 输出是 (B, num_classes)，用 accuracy 评估

# 分割 → task_type = "segmentation"
# 输出是 (B, C, H, W)，用 IoU 评估
```

## 步骤 2：创建任务目录

```bash
mkdir -p tasks/<task_name>
```

三个文件：

```
tasks/<task_name>/
├── __init__.py    # 声明默认模型和数据集
├── model.py       # 包装外部模型
└── dataset.py     # 包装数据加载
```

## 步骤 3：写 __init__.py

```python
DEFAULT_DATASET = "your_dataset_name"
DEFAULT_MODEL = "your_model_name"
```

这两个常量让 `train.py` 在未指定模型/数据集时有合理的默认值。

## 步骤 4：写 model.py

**模板（分类任务）：**

```python
import torch.nn as nn
from core.base_model import BaseModel
from core.registry import register_model

@register_model("your_model_name")
class YourModelWrapper(BaseModel):
    task_type = "classification"  # ← 声明任务类型

    def __init__(self, num_classes=10):
        super().__init__()
        # 导入并实例化外部模型
        from .external_model import ExternalModel
        self.net = ExternalModel(num_classes=num_classes)

    def forward(self, x):
        return self.net(x)

    def get_example_input(self):
        import torch
        return torch.randn(1, 3, 224, 224)  # ← 改成实际输入尺寸

    @classmethod
    def from_config(cls, config):
        return cls(num_classes=config.get("_num_classes", 10))
```

**模板（分割任务）：**

```python
import torch.nn as nn
from core.base_model import BaseModel
from core.registry import register_model

@register_model("your_model_name")
class YourSegModel(BaseModel):
    task_type = "segmentation"  # ← 关键区别

    def __init__(self):
        super().__init__()
        from .external_model import ExternalModel
        self.net = ExternalModel()

    def forward(self, x):
        # 如果数据集输出 5D (B,C,D,H,W) 但模型要 4D，
        # 在这里做 reshape 适配
        return self.net(x)

    def get_loss_fn(self):
        # 分割任务必须重写此方法
        return YourCustomLoss()

    def get_example_input(self):
        import torch
        return torch.randn(1, 1, 4, 512, 512)

    @classmethod
    def from_config(cls, config):
        return cls()
```

**容易踩的坑：**
- 输入形状不匹配：模型可能要 `(B, C, H, W)` 但数据集给了 `(B, C, D, H, W)`，在 `forward()` 里做 squeeze/reshape
- 忘记设 `task_type`：默认是 classification，分割任务会计算错误指标
- 忘记重写 `get_loss_fn()`：默认返回 CrossEntropyLoss，分割任务需要 DiceLoss 等

## 步骤 5：写 dataset.py

```python
from core.base_dataset import BaseDataset
from core.registry import register_dataset

@register_dataset("your_dataset_name")
class YourDataset(BaseDataset):
    CLASS_NAMES = ["class_0", "class_1", ...]  # 可选

    def __init__(self, data_dir, train=True, data_fraction=1.0):
        self.data_dir = data_dir
        self.train = train
        # 加载数据
        self.samples = self._load_data(data_dir, train)

        # --fast 模式：取子集
        if data_fraction < 1.0:
            n = max(1, int(len(self.samples) * data_fraction))
            self.samples = self.samples[:n]

    @property
    def num_classes(self):
        return len(self.CLASS_NAMES)  # 分割任务返回 2（背景+目标）

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 返回 (input, target)
        # 分类：return image, label
        # 分割：return frames, mask
        ...

    def get_preprocess_config(self):
        """返回推理预处理参数，export.py 会自动保存为 preprocess.json"""
        return {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "size": [224, 224],
            "channels": 3,
            "classes": self.CLASS_NAMES,
        }

    @classmethod
    def from_config(cls, config, split):
        return cls(
            data_dir=config["paths"]["data_dir"],
            train=(split == "train"),
            data_fraction=config.get("_data_fraction", 1.0),
        )
```

**from_config 的三个 split 值：**
- `"train"` → 训练集
- `"val"` → 验证集
- `"test"` → 测试集（evaluate.py 用）

## 步骤 6：复制依赖文件

如果外部项目有本地模块依赖（不是 pip 包的），复制到 `tasks/<task_name>/` 下：

```bash
cp /tmp/project/external_model.py tasks/<task_name>/
cp /tmp/project/utils.py tasks/<task_name>/
```

然后在 model.py/dataset.py 中用 **相对导入**：

```python
from .external_model import ExternalModel  # 正确
from external_model import ExternalModel   # 错误，找不到
```

## 步骤 7：更新 config.yaml

```yaml
project:
  name: "my-project"  # MLflow 实验名

full:
  model_name: "your_model_name"    # 和 @register_model 一致
  dataset_name: "your_dataset_name"  # 和 @register_dataset 一致
  epochs: 100
  batch_size: 32

# 分割任务额外加这段
segmentation:
  num_frame: 4
  img_size: 512
  train_split: "train.txt"
  val_split: "val.txt"
```

## 步骤 8：验证

```bash
# 1. 测试导入链
python -c "
import importlib
importlib.import_module('tasks.your_task.model')
importlib.import_module('tasks.your_task.dataset')
from core.registry import MODELS, DATASETS
print('Models:', list(MODELS.keys()))
print('Datasets:', list(DATASETS.keys()))
# 实例化模型
m = MODELS['your_model_name']()
print('task_type:', m.task_type)
print('loss:', type(m.get_loss_fn()).__name__)
# 前向传播
import torch
x = m.get_example_input()
with torch.no_grad():
    y = m(x)
print('Input:', list(x.shape), '→ Output:', list(y.shape))
"

# 2. 如果有真实数据，快速训练
python train.py --task your_task --fast
```

## 真实案例：e2e → sonar_detection

我们集成 LVNet（声纳多帧小目标检测）的完整过程：

**分析：**
- 模型：`LVNet(nn.Module)`，输入 `(B, C, D, H, W)`，输出 `(B, 1, D, H, W)`
- 任务：二值分割，用 DiceLoss + BCE
- 数据：txt 文件列出 DataRecord 文件夹，每文件夹含连续帧图像

**实现：**
- `task_type = "segmentation"`，`get_loss_fn()` 返回 `DiceFocalLoss`
- 数据集 `SonarFrameDataset` 从 txt 读取路径，堆叠 4 帧为输入
- 输入形状 `(1, 1, 4, 512, 512)`，模型参数量 1.92M
- 复制 `LVNet.py`、`muon.py`、`sonar_utils.py` 到任务目录

**结果：** `python train.py --task sonar_detection --fast` 可直接训练，MLflow 自动记录 IoU 指标。
