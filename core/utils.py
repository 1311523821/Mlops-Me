"""
工具函数：设备检测、配置加载、checkpoint 管理、数据指纹。
"""

import os
import hashlib
import json
from datetime import datetime
import yaml
import torch


def get_device(config: dict) -> torch.device:
    """
    自动检测并返回可用的计算设备。

    优先级: CUDA > CPU
    如果 config 中 use_cuda 为 False，则强制使用 CPU。

    参数:
        config: 配置字典

    返回:
        torch.device: 'cuda:0' 或 'cpu'
    """
    use_cuda = config.get("device", {}).get("use_cuda", True)
    if use_cuda and torch.cuda.is_available():
        gpu_id = config.get("device", {}).get("gpu_id", 0)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def load_config(config_path: str) -> dict:
    """
    加载 YAML 配置文件。

    参数:
        config_path: YAML 文件路径

    返回:
        dict: 配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(model: torch.nn.Module, optimizer, epoch: int,
                     metrics: dict, path: str):
    """
    保存模型 checkpoint，包含模型权重、优化器状态和训练元信息。

    参数:
        model:     模型实例
        optimizer: 优化器实例
        epoch:     当前 epoch 编号
        metrics:   指标字典，如 {"val_accuracy": 0.95}
        path:      保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model: torch.nn.Module, path: str,
                    device: torch.device = None) -> dict:
    """
    加载模型 checkpoint。

    参数:
        model:  模型实例（权重会被加载到其中）
        path:   checkpoint 文件路径
        device: 目标设备，None 则自动检测

    返回:
        dict: checkpoint 中的元信息（epoch, metrics 等）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return {
        "epoch": checkpoint.get("epoch", -1),
        "metrics": checkpoint.get("metrics", {}),
    }


def compute_data_fingerprint(dataset, num_samples: int = 100,
                             seed: int = 42) -> str:
    """
    计算数据集指纹 —— 对前 num_samples 个样本做 MD5 哈希。

    用途：追踪每次训练用了哪个版本的数据。
    数据不变 → 指纹不变；数据变了 → 指纹不同。
    这样在 MLflow 里可以对比不同数据版本的训练结果。

    参数:
        dataset:  PyTorch Dataset 实例
        num_samples: 采样数量（默认 100）
        seed:   随机种子（保证可复现）

    返回:
        str: 32 位 MD5 哈希字符串
    """
    import random
    random.seed(seed)

    n = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n)

    hasher = hashlib.md5()
    for idx in sorted(indices):
        data, _ = dataset[idx]
        # 取张量的字节表示做哈希
        if isinstance(data, torch.Tensor):
            hasher.update(data.numpy().tobytes())
        else:
            hasher.update(str(data).encode())

    return hasher.hexdigest()


def get_git_hash() -> str:
    """
    尝试获取当前 git commit 短哈希。
    如果不是 git 仓库或 git 不可用，返回 'unknown'。
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def build_run_name(task: str, fast: bool = False) -> str:
    """
    生成可读的 Run 名称，格式: 任务名-模式-时间戳

    参数:
        task: 任务名
        fast: 是否快速模式

    返回:
        str: 如 "demo-fast-20260522_142000"
    """
    mode = "fast" if fast else "full"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{task}-{mode}-{timestamp}"
