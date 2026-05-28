"""代码分析工具 — AST 类签名提取、依赖图、forward 追踪、硬编码检测。"""

import ast
import os
import re


def find_model_files(repo_path: str) -> list[str]:
    """
    在仓库中定位可能的模型文件。
    返回绝对路径列表，按可能性排序（文件名含 model/net/network 的优先）。
    """
    candidates = []
    model_keywords = ["model", "net", "network", "backbone", "encoder", "decoder"]
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "checkpoints", "outputs", "data"}]
        for f in files:
            if not f.endswith(".py"):
                continue
            full = os.path.join(root, f)
            lower = f.lower()
            score = sum(2 for kw in model_keywords if kw in lower)
            candidates.append((score, full))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates]


def extract_init_params(file_path: str, class_name: str = None) -> dict:
    """
    从 Python 文件中提取指定类（或第一个 nn.Module 子类）的 __init__ 参数。
    返回 {"class_name": str, "params": {"param_name": default_value, ...}}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)

    target_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if class_name and node.name == class_name:
                target_class = node
                break
            if not class_name and _is_module_subclass(node):
                target_class = node
                break

    if target_class is None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                target_class = node
                break

    if target_class is None:
        return {"class_name": None, "params": {}}

    init_method = None
    for item in target_class.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            init_method = item
            break

    if init_method is None:
        return {"class_name": target_class.name, "params": {}}

    params = {}
    args = init_method.args
    all_args = args.args[1:] if args.args and args.args[0].arg == "self" else args.args
    defaults = args.defaults
    num_no_default = len(all_args) - len(defaults)
    for i, arg in enumerate(all_args):
        if i >= num_no_default:
            d = defaults[i - num_no_default]
            try:
                val = ast.literal_eval(d)
            except (ValueError, SyntaxError):
                val = ast.unparse(d) if hasattr(ast, "unparse") else str(d)
            params[arg.arg] = val
        else:
            params[arg.arg] = None

    return {"class_name": target_class.name, "params": params}


def find_dataset_files(repo_path: str) -> list[str]:
    """在仓库中定位可能的数据集文件。"""
    candidates = []
    dataset_keywords = ["dataset", "data_loader", "dataloader", "data"]
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        for f in files:
            if not f.endswith(".py"):
                continue
            full = os.path.join(root, f)
            lower = f.lower()
            score = sum(2 for kw in dataset_keywords if kw in lower)
            candidates.append((score, full))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates]


def detect_dependencies(repo_path: str) -> dict:
    """
    分析仓库依赖。返回 {"pip": [...], "local": [...]}
    pip: requirements.txt 或 setup.py 中的包名
    local: 仓库内非 pip 的 .py 模块
    """
    result = {"pip": [], "local": []}

    req_path = os.path.join(repo_path, "requirements.txt")
    if os.path.isfile(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            result["pip"] = [
                line.strip().split("==")[0].split(">=")[0].strip()
                for line in f if line.strip() and not line.startswith("#")
            ]

    setup_path = os.path.join(repo_path, "setup.py")
    if os.path.isfile(setup_path):
        with open(setup_path, "r", encoding="utf-8") as f:
            content = f.read()
            found = re.findall(r"['\"]([\w-]+)[\"']\s*[,\)]", content)
            result["pip"].extend(found)

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "tests", "test"}]
        for f in files:
            if f.endswith(".py") and not f.startswith("test_"):
                rel = os.path.relpath(os.path.join(root, f), repo_path)
                result["local"].append(rel)

    result["pip"] = sorted(set(result["pip"]))
    return result


def detect_hardcoded_values(file_path: str) -> list[dict]:
    """
    检测文件中的潜在硬编码值（数字和字符串字面量）。
    返回 [{"value": ..., "line": int, "context": str}, ...]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    hardcoded = []
    suspicious_patterns = [
        (r"(?<!\w)\d{2,}(?!\w)", "数字常量", 0),
        (r"path\s*=\s*([\"'][^\"']+[\"'])", "路径字符串", 1),
        (r"size\s*=\s*(\d+)", "尺寸参数", 1),
    ]

    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()
        if line_stripped.startswith("#") or line_stripped.startswith('"""'):
            continue
        for pattern, desc, group_idx in suspicious_patterns:
            m = re.search(pattern, line_stripped)
            if m:
                hardcoded.append({
                    "value": m.group(group_idx),
                    "line": i,
                    "context": line_stripped[:80],
                    "type": desc,
                })

    return hardcoded


def _is_module_subclass(node: ast.ClassDef) -> bool:
    """判断 AST 类节点是否继承自 nn.Module。"""
    for base in node.bases:
        if isinstance(base, ast.Attribute):
            if base.attr == "Module" and _get_name(base.value) in ("nn", "torch.nn"):
                return True
        if isinstance(base, ast.Name) and base.id == "Module":
            return True
        if isinstance(base, ast.Call):
            return False
    return False


def _get_name(node: ast.AST) -> str:
    """获取 AST 节点的字符串表示。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_get_name(node.value)}.{node.attr}"
    return ""
