"""验证工具 — 导入链、前向传播、快速训练验证。"""

import subprocess
import sys
import os
from pathlib import Path


def validate_imports(project_root: str) -> dict:
    """
    验证导入链：检查 task 注册到 MODELS/DATASETS。
    返回 {"pass": bool, "model_name": str, "dataset_name": str, "error": str|None}
    """
    task_name = os.path.basename(os.getcwd())
    code = f'''
import sys
sys.path.insert(0, r"{project_root}")
from core.registry import MODELS, DATASETS
# 重新加载模块触发注册
import importlib
for mod in list(sys.modules.keys()):
    if "tasks." in mod:
        del sys.modules[mod]
try:
    importlib.import_module("tasks.{task_name}.model")
    importlib.import_module("tasks.{task_name}.dataset")
except Exception as e:
    print(f"IMPORT_ERROR: {{e}}")
    raise SystemExit(1)
print(f"MODELS: {{list(MODELS.keys())}}")
print(f"DATASETS: {{list(DATASETS.keys())}}")
'''
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=project_root,
    )
    if r.returncode != 0:
        return {"pass": False, "model_name": None, "dataset_name": None, "error": r.stderr or r.stdout}
    return {"pass": True, "model_name": None, "dataset_name": None, "error": None}


def validate_forward_pass(project_root: str, model_name: str) -> dict:
    """
    验证前向传播：创建模型 → get_example_input() → forward() → 检查输出 shape。
    返回 {"pass": bool, "input_shape": list, "output_shape": list, "error": str|None}
    """
    code = f'''
import sys
sys.path.insert(0, r"{project_root}")
import importlib
importlib.import_module("tasks.{model_name}.model")
from core.registry import MODELS
import torch

m = MODELS["{model_name}"]()
x = m.get_example_input()
print(f"INPUT: {{list(x.shape)}}")
with torch.no_grad():
    y = m(x)
print(f"OUTPUT: {{list(y.shape)}}")
print(f"TASK_TYPE: {{m.task_type}}")
'''
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=project_root,
    )
    if r.returncode != 0:
        return {"pass": False, "input_shape": None, "output_shape": None, "error": r.stderr}
    return {"pass": True, "input_shape": [], "output_shape": [], "error": None}


def validate_train_fast(project_root: str, task_name: str) -> dict:
    """
    验证快速训练：python train.py --task {task_name} --fast。
    返回 {"pass": bool, "stdout": str, "error": str|None}
    """
    r = subprocess.run(
        [sys.executable, "train.py", "--task", task_name, "--fast"],
        capture_output=True, text=True, cwd=project_root, timeout=300,
    )
    if r.returncode != 0:
        return {"pass": False, "stdout": r.stdout, "error": r.stderr}
    return {"pass": True, "stdout": r.stdout, "error": None}


def run_all_validations(project_root: str, task_name: str, model_name: str) -> dict:
    """依次运行全部验证步骤。"""
    results = {}

    results["import"] = validate_imports(project_root)
    if not results["import"]["pass"]:
        results["status"] = "failed"
        return results

    results["forward"] = validate_forward_pass(project_root, model_name)
    if not results["forward"]["pass"]:
        results["status"] = "failed"
        return results

    results["train_fast"] = validate_train_fast(project_root, task_name)
    results["status"] = "passed" if results["train_fast"]["pass"] else "partial"

    return results
