"""
训练入口脚本。

用法:
    python train.py --task demo                      # 全量训练
    python train.py --task demo --fast               # 快速验证模式
    python train.py --task demo --config my_cfg.yaml  # 使用自定义配置

快速模式 vs 全量模式:
    --fast:  10% 数据、3 epoch、轻量模型（simple_cnn）
    不加:    100% 数据、50 epoch、完整模型（resnet18）
"""

import argparse
import importlib
import os
from torch.utils.data import DataLoader
from ruamel.yaml import YAML
from core.utils import load_config, get_device, compute_data_fingerprint, get_git_hash, build_run_name
from core.registry import MODELS, DATASETS
from core.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="MLOps 训练脚本")
    parser.add_argument("--task", type=str, required=True,
                        help="任务名，对应 tasks/ 下的目录名（如 demo）")
    parser.add_argument("--fast", action="store_true",
                        help="启用快速验证模式")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="全局配置文件路径")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断的 checkpoint 恢复训练")
    args = parser.parse_args()

    # ---------- 1. 加载配置 ----------
    config = load_config(args.config)

    # 传递任务特定参数（分割任务的帧数、图像尺寸等）
    seg_cfg = config.get("segmentation", {})
    if seg_cfg:
        config.setdefault("_num_frame", seg_cfg.get("num_frame", 4))
        config.setdefault("_img_size", seg_cfg.get("img_size", 512))
        config.setdefault("_train_split", seg_cfg.get("train_split", "train1.txt"))
        config.setdefault("_val_split", seg_cfg.get("val_split", "val_new.txt"))
        config.setdefault("_test_split", seg_cfg.get("test_split", "val_new.txt"))

    # ---------- 2. --fast 模式覆盖参数 ----------
    if args.fast:
        fast_cfg = config.get("fast", {})
        config["_data_fraction"] = fast_cfg.get("data_fraction", 0.1)
        config["_epochs"] = fast_cfg.get("epochs", 3)
        config["_model_name"] = fast_cfg.get("model_name", "simple_cnn")
        config["_batch_size"] = fast_cfg.get("batch_size", 32)
        config["_dataset_name"] = fast_cfg.get("dataset_name", "")
        mode_str = "快速模式"
    else:
        full_cfg = config.get("full", {})
        config["_data_fraction"] = full_cfg.get("data_fraction", 1.0)
        config["_epochs"] = full_cfg.get("epochs", 50)
        config["_model_name"] = full_cfg.get("model_name", "resnet18")
        config["_batch_size"] = full_cfg.get("batch_size", 64)
        config["_dataset_name"] = full_cfg.get("dataset_name", "")
        mode_str = "全量模式"

    config["_task_name"] = args.task

    # ---------- 3. 导入 tasks/<task> 包，触发模型和数据集的注册 ----------
    try:
        importlib.import_module(f"tasks.{args.task}.model")
        importlib.import_module(f"tasks.{args.task}.dataset")
    except ModuleNotFoundError as e:
        print(f"错误: 找不到任务模块 'tasks.{args.task}'")
        print(f"请确保 tasks/{args.task}/ 目录下有 model.py 和 dataset.py")
        print(f"原始错误: {e}")
        return

    # ---------- 4. 读任务包声明的默认数据集/模型 ----------
    try:
        task_pkg = importlib.import_module(f"tasks.{args.task}")
        task_default_dataset = getattr(task_pkg, "DEFAULT_DATASET", "")
        task_default_model = getattr(task_pkg, "DEFAULT_MODEL", "")
    except ModuleNotFoundError:
        task_default_dataset = ""
        task_default_model = ""

    # ---------- 5. 从注册表取模型和数据集 ----------
    model_name = config["_model_name"]
    if model_name not in MODELS:
        # 先用任务包声明的默认模型，再不行就用第一个已注册的
        if task_default_model and task_default_model in MODELS:
            model_name = task_default_model
        else:
            available = list(MODELS.keys())
            if not available:
                print("错误: 没有注册任何模型")
                return
            model_name = available[0]
        print(f"提示: 模型 '{config['_model_name']}' 未注册，自动选用 '{model_name}'")
        config["_model_name"] = model_name

    # 数据集选择：config 指定 > 任务包默认 > 第一个已注册的
    dataset_name = config.get("_dataset_name") or task_default_dataset or list(DATASETS.keys())[0]
    if dataset_name not in DATASETS:
        print(f"错误: 未找到数据集 '{dataset_name}'")
        print(f"已注册的数据集: {list(DATASETS.keys())}")
        return

    # ---------- 5. 创建数据集和数据加载器 ----------
    train_dataset = DATASETS[dataset_name].from_config(config, split="train")
    val_dataset = DATASETS[dataset_name].from_config(config, split="val")

    config["_num_classes"] = train_dataset.num_classes
    config["_dataset_name"] = dataset_name

    # 数据指纹：追踪数据集版本
    print("计算数据指纹...")
    data_fingerprint = compute_data_fingerprint(train_dataset)
    config["_data_fingerprint"] = data_fingerprint
    print(f"数据指纹 (MD5): {data_fingerprint}")

    # Git 版本
    git_hash = get_git_hash()
    config["_git_hash"] = git_hash
    if git_hash != "unknown":
        print(f"Git commit: {git_hash}")

    # Run 名称
    config["_run_name"] = build_run_name(args.task, args.fast)

    batch_size = config["_batch_size"]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    # ---------- 6. 创建模型 ----------
    model = MODELS[model_name].from_config(config)

    # ---------- 7. 打印信息，开始训练 ----------
    device = get_device(config)
    print(f"{'='*60}")
    print(f"任务: {args.task}  |  模型: {model_name}")
    print(f"数据集: {dataset_name}  |  类别数: {train_dataset.num_classes}")
    print(f"设备: {device}  |  模式: {mode_str}")
    print(f"训练集大小: {len(train_dataset)}  |  批大小: {batch_size}")
    print(f"Epochs: {config['_epochs']}")
    print(f"{'='*60}")

    # 断点续训路径
    resume_path = None
    if args.resume:
        output_dir = config.get("paths", {}).get("output_dir", "./outputs")
        resume_path = os.path.join(output_dir, args.task, "best_model.pth")
        if not os.path.exists(resume_path):
            print(f"警告: 未找到 checkpoint {resume_path}，将从头训练")
            resume_path = None

    trainer = Trainer(model, train_loader, val_loader, config,
                      resume_from=resume_path)
    trainer.train()


if __name__ == "__main__":
    main()
