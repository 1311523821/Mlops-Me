---
name: paper2mlops
description: |
  把论文（arXiv）、GitHub 仓库、博客文章等任意 ML 资料，
  通过 Agent 自动分析并生成符合 MLOps 框架规范的 tasks/<name>/ 代码。
  当用户说 /paper2mlops、或提到把论文/仓库接入 MLOps 时触发。
---

# Paper2MLOps — 论文/代码自动接入 MLOps

## 核心原则

- **不要硬编码参数**：所有配置值都从原代码分析出来，MLOps 只做包装不做假设。参数名、参数值、输入形状、预处理参数 — 一切以原项目为准
- **Agent = Model + Harness**：Model 通过 Claude API 调用，Harness 是 `core/paper2mlops/` 提供的工具层
- **全部本地化**：下载、分析、代码生成全在本地，HTTP 服务器仅绑定 127.0.0.1
- **半自动**：分析报告需用户确认后才生成代码；审查报告需用户确认后才完成
- **测试先行 + 审查**：代码生成前先写测试；生成后跑 Agent-to-Agent 审查
- **每次失败 → 工程化修复**：Agent 出错不是手动修，而是改进 Harness 约束规则

## 流程概览

```
输入获取&分类 → 多Agent并行分析 → 分析报告+人工确认 → 测试先行→代码生成→审查→验证
```

| 旧流程 (add-mlops-project) | 新流程 (paper2mlops) |
|---|---|
| 分析项目 | **Harness 自动获取**：下载论文/clone 仓库/抓取网页 |
| （无） | **多 Agent 并行分析**：Paper Agent + Code Agent 同时工作 |
| （无） | **HTML 报告 + 人工确认**：在线查看、修改、确认分析结果 |
| 写 model.py → dataset.py | **测试先行**：先生成测试用例，再生成代码 |
| 复制依赖 → config.yaml | **代码生成**：model.py + dataset.py + config.yaml + 依赖复制 |
| （手动审查） | **Agent-to-Agent 自动审查**：独立 subagent 六项清单检查 |
| 验证 | **三层验证**：导入链 → 前向传播 → train --fast |

## 工具清单

以下 Python 工具通过 Bash 调用，均在 `core/paper2mlops/` 下：

| 工具 | 用途 | 调用示例 |
|------|------|---------|
| harvester.py | 输入获取和分类 | `python -c "from core.paper2mlops.harvester import ..."` |
| parser.py | 论文 PDF 解析 | `python -c "from core.paper2mlops.parser import ..."` |
| analyzer.py | 代码 AST 分析 | `python -c "from core.paper2mlops.analyzer import ..."` |
| reports.py | 报告生成和渲染 | `python -c "from core.paper2mlops.reports import ..."` |
| server.py | 本地 HTTP 服务器 | `python -c "from core.paper2mlops.server import ..."` |
| generator.py | 任务代码生成 | `python -c "from core.paper2mlops.generator import ..."` |
| validator.py | 验证测试 | `python -c "from core.paper2mlops.validator import ..."` |

## 四阶段工作流

### 阶段一：输入获取 & 分类

用户运行 `/paper2mlops <arg1> <arg2> ...`，Dispatcher 执行：

```bash
python -c "
from core.paper2mlops.harvester import detect_inputs
import json
result = detect_inputs(<用户的所有参数>)
print(json.dumps(result, ensure_ascii=False))
"
```

根据 `detect_inputs` 的结果判断：
- 有 paper → 需要 Paper Agent
- 有 code → 需要 Code Agent
- 两者都有 → 并行分发

**获取资料（按需）：**

论文下载：
```bash
python -c "
from core.paper2mlops.harvester import fetch_paper
import json
r = fetch_paper('<source>', '.superpowers/paper2mlops/<task_id>/raw')
print(json.dumps(r, ensure_ascii=False))
"
```

代码克隆：
```bash
python -c "
from core.paper2mlops.harvester import fetch_code
import json
r = fetch_code('<source>', '.superpowers/paper2mlops/<task_id>/raw')
print(json.dumps(r, ensure_ascii=False))
"
```

---

### 阶段二：多 Agent 并行分析

**如输入包含论文 → 启动 Paper Agent（Subagent）：**

```
Agent(
  subagent_type="general-purpose",
  description="分析论文",
  prompt="你是论文分析 Agent。分析以下论文，提取结构化信息。

论文 PDF 路径：{pdf_path}

请用 core/paper2mlops/parser.py 的工具函数提取：
1. 论文文本（extract_text）
2. 模型架构描述段落（find_architecture_section）— 重点读 Method/Architecture 章节
3. GitHub 链接（find_github_links）— 搜索论文中提到的官方实现
4. Loss 函数定义（find_loss_function）— 识别训练用的损失函数类型
5. 输入形状（find_input_shape）— 从论文描述中提取输入尺寸
6. 预处理参数（find_preprocess_params）— 提取 mean/std 等归一化参数

输出一个 JSON 对象，包含以上全部字段。如有 GitHub 链接，标注出来供后续 Code Agent 使用。

关键原则：不要猜测，所有值从论文原文提取。不确定的字段标注为 null，不要编造。"
)
```

**如输入包含代码 → 启动 Code Agent（Subagent）：**

```
Agent(
  subagent_type="general-purpose",
  description="分析代码仓库",
  prompt="你是代码分析 Agent。分析以下仓库，提取结构化信息。你的任务是回答以下每个问题，全部从原代码中提取，不做猜测。

仓库路径：{repo_path}

## 分析清单

| 问题 | 去哪里找 | 怎么确定 |
|------|----------|----------|
| 模型类名和构造函数参数？ | class XXX(nn.Module) | 看 __init__ 的签名，每个参数的值从默认值或 argparse 中取 |
| 输入形状？ | forward() 第一行 + 训练循环 | 打印第一个 batch 的 shape，或看 Dataset 的 __getitem__ |
| 输出形状？ | forward() 最后一行 | print(output.shape) |
| 任务类型？ | 损失函数 | CrossEntropyLoss → classification；BCE/Dice/IoU → segmentation |
| 数据格式？ | Dataset 类或训练脚本 | txt/json/文件夹结构 → 决定 dataset.py 怎么读 |
| 本地依赖？ | import 语句 | 非 pip 安装的 .py 文件都要复制 |
| 预处理参数？ | Dataset 的 transform | mean、std、resize 尺寸，全部从原代码提取 |

具体步骤：

1. 用 find_model_files() 定位模型文件，按文件名相关度排序
2. 用 extract_init_params() 提取 __init__ 的参数名和默认值
   - 如果类继承 nn.Module，自动找到；否则指定类名
   - 返回的 params 中，值为 None 的是必填参数（无默认值）
3. 用 find_dataset_files() 定位数据集文件
4. 用 detect_dependencies() 识别 pip 包和本地 .py 依赖
5. 用 detect_hardcoded_values() 检测可能硬编码的数字/路径

输出 JSON：
{
  "model": {"class_name": "...", "source_file": "...", "init_params": {...}, "input_shape": [...], "output_shape": [...]},
  "dataset": {"format": "...", "num_classes": N, "class_names": [...]},
  "task_type": "classification|segmentation",
  "dependencies": {"pip": [...], "local_files": [...]},
  "hardcoded": [...]
}

关键原则：不要猜测，全部从原代码中复制。不确定的值标注为 null。参数名跟原项目保持一致，不做翻译。"
)
```

**如果两个 Agent 都可用，并行启动它们（同一条消息中两个 Agent 工具调用）。**

---

### 阶段三：分析报告 + 人工确认

收集子 Agent 输出，合并生成报告：

```bash
python -c "
from core.paper2mlops.reports import build_report_json, render_report_html
import json

paper = json.loads('''<Paper Agent 的 JSON 输出>''') if '<有论文>' else None
code = json.loads('''<Code Agent 的 JSON 输出>''') if '<有代码>' else None

report = build_report_json(
    task_name='<推断的任务名>',
    source={'paper_url': '...', 'github_url': '...', 'input_type': 'paper+code'},
    paper_analysis=paper,
    code_analysis=code,
)
# 保存 JSON
with open('.superpowers/paper2mlops/<task_id>/analysis/report.json', 'w') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
# 渲染 HTML
html = render_report_html(report)
with open('.superpowers/paper2mlops/<task_id>/analysis/report.html', 'w') as f:
    f.write(html)
"
```

**启动 HTTP 服务器（如未运行）：**

```bash
python -c "
from core.paper2mlops.server import start_server
start_server(
    report_dir='.superpowers/paper2mlops/<task_id>/analysis',
    state_dir='.superpowers/paper2mlops/<task_id>/state',
)
print('报告已生成: http://localhost:65049')
"
```

**等待用户确认：**

```bash
python -c "
from core.paper2mlops.server import check_confirmation
import json
event = check_confirmation('.superpowers/paper2mlops/<task_id>/state', timeout_seconds=600)
print(json.dumps(event) if event else 'TIMEOUT')
"
```

如果用户修改了字段，更新 report.json 中的对应值后再继续。

---

### 阶段四：测试先行 → 代码生成 → 代码审查

**Step 0: 阅读参考章节**

在生成代码之前，必须阅读本文件末尾的以下参考章节：
- **model.py 编写规范** — task_type 对照表、常见坑
- **dataset.py 编写规范** — 数据格式分析方法
- **config.yaml 编写原则** — 字段名不做翻译
- **依赖处理** — sys.path 修复技巧

**Step 1: 生成代码**

```bash
python -c "
from core.paper2mlops.generator import generate_task, copy_dependencies
import json

with open('.superpowers/paper2mlops/<task_id>/analysis/report.json') as f:
    report = json.load(f)

files = generate_task(report['task_name'], report, '.')
print('Generated:', files)
"
```

如有本地依赖文件，复制它们。**注意处理绝对导入**：如果外部模块内部有 `from LVNet import xxx` 这类绝对导入，在 model.py 的 `__init__` 开头加入 sys.path 修复：

```python
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from ExternalModule import ExternalModel
```

```bash
python -c "
from core.paper2mlops.generator import copy_dependencies
import json
with open('.superpowers/paper2mlops/<task_id>/analysis/report.json') as f:
    report = json.load(f)
copied = copy_dependencies(report, '<repo_path>', report['task_name'], '.')
print('Copied:', copied)
"
```

**Step 2: 运行验证**

先跑精确的导入链 + 前向传播测试：

```bash
python -c "
import importlib
importlib.import_module('tasks.<task_name>.model')
importlib.import_module('tasks.<task_name>.dataset')
from core.registry import MODELS, DATASETS
m = MODELS['<model_name>']()
print('task_type:', m.task_type)
print('loss:', type(m.get_loss_fn()).__name__)
import torch
x = m.get_example_input()
with torch.no_grad():
    y = m(x)
print('Input:', list(x.shape), '→ Output:', list(y.shape))
"
```

通过后跑快速训练验证（如有数据）：

```bash
python train.py --task <task_name> --fast
```

如果验证失败：分析错误信息 → 修正代码 → 重新验证。最多循环 3 次。

**Step 3: 代码审查（确认点 2）**

启动 Reviewer Agent（Subagent），独立审查：

```
Agent(
  subagent_type="feature-dev:code-reviewer",
  description="审查新生成的 MLOps 任务",
  prompt="审查 tasks/<task_name>/ 下所有文件是否合规。按以下清单逐项检查，报告通过/不通过：

1. 低耦合：core/ 不包含具体任务名
2. 硬编码：参数来自 config 而非写死
3. 接口合规：继承 BaseModel/BaseDataset，实现必要方法
4. 安全隐患：无密钥泄露，gitignore 完整
5. 代码质量：简体中文注释，语义清晰命名
6. 测试验证：三层测试通过"
)
```

审查结果渲染为 HTML 审查报告：
```bash
python -c "
from core.paper2mlops.reports import render_review_html
review = <审查结果 JSON>
html = render_review_html(review)
with open('.superpowers/paper2mlops/<task_id>/analysis/review.html', 'w') as f:
    f.write(html)
"
```

等待用户确认审查结果。如有 ❌ 项，自动修复后重新审查。

**Step 4: 快速训练验证（如有数据）**

```bash
python train.py --task <task_name> --fast
```

**Step 5: 最终结果确认（确认点 3）**

展示生成的文件清单和验证结果。用户确认后完成。

---

## 错误处理

- 下载失败 → 提示用户检查链接，或手动提供文件
- Agent 信心度 low → HTML 报告中红色高亮，要求用户手动填写
- 测试不通过 → 分析错误信息 → 自动修正 → 最多 3 次循环 → 停止并报告
- 审查不通过 → 自动修复 → 最多 3 次循环 → 停止并展示问题清单

## 持久化路径

所有中间产物写入 `.superpowers/paper2mlops/{task_id}/`：
- `raw/` — 下载的论文 PDF、克隆的仓库
- `analysis/` — 分析报告 JSON + HTML
- `state/` — 用户确认事件
- `review.md` — 代码审查报告

---

## 参考：model.py 编写规范

Generator Agent 生成 model.py 时，必须遵循以下规范。

### task_type 对照表

| 原代码用的 loss | task_type | get_loss_fn 返回 |
|----------------|-----------|-----------------|
| CrossEntropyLoss | `"classification"` | 不需要重写（默认） |
| BCEWithLogitsLoss | `"segmentation"` | DiceLoss 或自定义 |
| DiceLoss / FocalLoss | `"segmentation"` | 原项目的 loss 类 |

### 编写三步法

**第一步：从原代码提取构造函数参数** — 找到 `__init__` 签名，每个参数抄下来作为 wrapper 的 `__init__` 参数，默认值保持一致。

**第二步：确定 task_type** — 根据上表从 loss 函数推断。

**第三步：确定输入形状** — 从 Dataset `__getitem__` 或训练循环中获取 batch shape，写入 `get_example_input()`。

### 模板

```python
import torch.nn as nn
from core.base_model import BaseModel
from core.registry import register_model

@register_model("从原项目的模型名推断")
class YourModelWrapper(BaseModel):
    task_type = "从原项目的 loss 推断"

    def __init__(self, <从原代码 __init__ 签名复制的参数>):
        super().__init__()
        # 导入外部模型（注意处理本地导入路径问题，见依赖处理章节）
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
        # 从 config 中读取参数。键名从原代码 argparse 参数名推断
        return cls(<参数映射>)
```

### 容易踩的坑

- **输入形状不匹配**：模型要 4D 但数据集给了 5D → 在 `forward()` 里做 reshape
- **忘记设 `task_type`**：默认 classification，分割任务会算错指标
- **忘记重写 `get_loss_fn()`**：默认 CrossEntropyLoss 不能用于分割
- **外部模型的本地 import 路径**：见依赖处理章节的 sys.path 修复

---

## 参考：dataset.py 编写规范

### 分析原项目的数据格式

仔细看原项目的 Dataset 类或训练脚本，回答：
- 数据文件类型？（图片/txt/npy/...）
- 目录结构？（文件夹分类 / txt 标注路径 / json 标注 / ...）
- 标签格式？（整数类别 / 二值掩码 / bbox / ...）
- 预处理参数？（mean、std、resize 尺寸 — **全部从原代码提取**）

### 模板

```python
from core.base_dataset import BaseDataset
from core.registry import register_dataset

@register_dataset("从原项目推断的名字")
class YourDataset(BaseDataset):
    CLASS_NAMES = <从原代码提取的类别列表>

    def __init__(self, data_dir, train=True, data_fraction=1.0,
                 <从原代码提取的其他参数>):
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

---

## 参考：config.yaml 编写原则

**不要改全局 `config.yaml`**。创建 `tasks/<task_name>/config.yaml`。

**原则：config 里的字段名反映原项目的参数名，不做翻译。原代码叫什么，config 里就叫什么。**

配置合并规则：`tasks/<name>/config.yaml` 覆盖全局 `config.yaml` 同名字段。`train.py`/`evaluate.py`/`export.py` 自动合并。

### 必需段

```yaml
fast:
  model_name: "和 @register_model 一致"
  dataset_name: "和 @register_dataset 一致"
  epochs: <快速验证，通常 3>
  data_fraction: 0.1

full:
  model_name: "和 @register_model 一致"
  dataset_name: "和 @register_dataset 一致"
  epochs: <原项目默认 epoch 数>
```

### 按需添加的段

```yaml
# 模型构造参数（如有，字段名跟原 argparse 一致）
model_params:
  num_classes: 1000
  dropout: 0.5

# 数据集参数（如有）
dataset_params:
  image_size: 224

# 损失函数参数（如有自定义 loss）
loss_params:
  alpha: 0.25
  gamma: 2.0
```

---

## 参考：依赖处理

### 复制本地文件

检查原项目的 import 语句，把非 pip 包的本地 `.py` 文件复制到 `tasks/<task_name>/` 下。

### sys.path 修复

如果外部模块内部有绝对导入（如 `from LVNet import xxx`），在 `model.py` 的 `__init__` 开头加入：

```python
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from ExternalModule import ExternalModel
```

---

## 参考：真实案例 — sonar_detection

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
