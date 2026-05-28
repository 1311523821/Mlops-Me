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

## 代码审查

每次提交或合并前，**必须用 subagent 做代码审查**。审查员独立检查代码，不和实现者共享上下文，避免"盲点传染"。

### 审查时机
- 新任务完成 → 审查 `tasks/<name>/` 全部文件
- 修改 `core/` → 审查 core/ + 跑 demo 验证兼容
- 提交前 → 审查本次 diff 中的所有文件

### 审查标准（subagent 用以下 prompt）

```
审查本次改动，按以下清单逐项检查，报告通过 / 不通过的项：

1. 低耦合检查
   - core/ 是否包含具体任务名、模型名、数据集名？（有则违规）
   - tasks/xxx/model.py 是否只定义模型，不包含训练循环？（包含则违规）
   - 入口脚本是否只做编排，不写业务逻辑？

2. 硬编码检查
   - 模型参数是否从 config 读取而非写死在代码里？
   - 预处理参数（mean/std/size）是否从原代码提取而非编造？
   - task_type、loss 函数是否正确声明？

3. 接口合规
   - model.py：是否继承 BaseModel、实现 get_example_input()、from_config()？
   - dataset.py：是否继承 BaseDataset、实现 num_classes、from_config()？
   - 分割任务是否重写 get_loss_fn()？
   - 是否用 @register_model / @register_dataset 装饰器注册？

4. 安全隐患
   - 是否有硬编码的 token、密码、API key、内网 IP？
   - gitignore 是否覆盖 data/、outputs/、mlflow.db、.env？

5. 测试验证
   - 新任务的导入链和前向传播是否通过？
   - 改动 core/ 后 demo 任务是否向后兼容？

6. 代码质量
   - 注释是否用简体中文？
   - 变量命名是否语义清晰？
```

### 使用方式

```
# 提交前审查
Agent(subagent_type="feature-dev:code-reviewer", description="审查本次改动",
prompt="审查本次 diff 中的改动，按 CLAUDE.md 的审查清单逐项检查。")

# 新任务审查
Agent(subagent_type="feature-dev:code-reviewer", description="审查新任务 xxx",
prompt="审查 tasks/xxx/ 下所有文件是否合规。审查清单见 CLAUDE.md。")
```

审查报告中有任何 ❌ 项，先修复再提交。
