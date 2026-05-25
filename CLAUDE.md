# CLAUDE.md — MLOps 工作流项目指令

## 核心原则

- **低耦合**：core/ 不依赖任何具体任务；tasks/ 只实现模型和数据，不处理训练循环；入口脚本只做编排
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
