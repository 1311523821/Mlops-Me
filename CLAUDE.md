# CLAUDE.md — MLOps 工作流项目指令

## 核心原则

- **低耦合**：core/ 不依赖任何具体任务；tasks/ 只实现模型和数据，不处理训练循环；入口脚本只做编排
- **测试先行**：开发新功能、新任务、或修改 core/ 之前，先设计测试用例。写清楚"验证什么、怎么验证、预期结果"，再动手实现
- **中文回答**：所有回复使用简体中文
- **中文注释**：所有代码注释使用简体中文

## 项目结构

```
core/          # 框架无关基础设施（不应包含任何具体模型名或数据集名）
tasks/         # 具体任务（每个任务 = model.py + dataset.py）
train.py       # 入口编排器
evaluate.py    # 评估
export.py      # ONNX
deploy/        # 推理服务
```

## 抽象接口

- `BaseModel` → `get_example_input()`, `from_config()`
- `BaseDataset` → `num_classes`, `from_config()`, `get_preprocess_config()`
- `Trainer` → 接收 model + dataloader + config，不关心具体任务
- `registry` → 装饰器注册，字典查找

## 命令

```bash
python train.py --task demo --fast          # 快速验证
python train.py --task demo                 # 全量训练
python train.py --task demo --resume        # 断点续训
python evaluate.py --task demo              # 评估
python export.py --task demo                # 导出 ONNX
python deploy/app.py --model outputs/demo/model.onnx  # 推理服务
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

## 测试规范

开发新功能前，先设计测试用例。每次测试覆盖三层：

### 1. 导入和注册链
```python
importlib.import_module('tasks.xxx.model')
importlib.import_module('tasks.xxx.dataset')
assert '模型名' in MODELS
assert '数据集名' in DATASETS
```

### 2. 前向传播
```python
m = MODELS['模型名'].from_config(config)
x = m.get_example_input()
y = m(x)
assert y.shape 符合预期
```

### 3. 完整流程（如有数据）
```bash
python train.py --task xxx --fast   # 快速训练不报错
python evaluate.py --task xxx       # 评估生成图表
python export.py --task xxx         # ONNX 导出 + preprocess.json
```

### 测试用例设计原则
- 每个新任务的 `model.py` 和 `dataset.py` 写完后立即跑导入 + 前向测试
- 改动 `core/` 后跑 `demo` 确保向后兼容
- 测试用例放在任务描述中或 `tests/` 目录下
- 测试失败不继续后续步骤，先修复再推进
