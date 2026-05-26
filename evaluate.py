"""
评估脚本。

用法:
    python evaluate.py --task demo

执行流程:
    1. 加载 best_model.pth
    2. 在测试集上评估，打印指标
    3. 通过 MLflow 查历史最佳准确率
    4. 生成对比柱状图和混淆矩阵
"""

import argparse
import importlib
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，服务器上也能用
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import mlflow

from core.utils import load_merged_config, get_device, load_checkpoint
from core.registry import MODELS, DATASETS


def plot_comparison(current_acc: float, best_historical: float,
                    task_name: str, output_dir: str):
    """
    绘制当前模型 vs 历史最佳的对比柱状图。
    """
    labels = ["Current Model", "Best Historical"]
    values = [current_acc, best_historical]
    colors = ["#4CAF50", "#2196F3"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, width=0.4)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Model Comparison - {task_name}")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"对比图已保存: {path}")
    return path


def plot_confusion_matrix(y_true: list, y_pred: list, class_names: list,
                          task_name: str, output_dir: str):
    """
    绘制混淆矩阵热力图。
    """
    cm = confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.8),
                                     max(6, num_classes * 0.6)))
    im = ax.imshow(cm, cmap="Blues")

    # 在每个格子中显示数字
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {task_name}")
    fig.colorbar(im, ax=ax)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"混淆矩阵已保存: {path}")
    return path


def get_best_historical_accuracy(task_name: str, mlflow_uri: str) -> float:
    """
    从 MLflow 历史记录中查找该任务的最高准确率。

    如果找不到历史记录（第一次跑），返回 0.0。
    """
    mlflow.set_tracking_uri(mlflow_uri)
    try:
        experiment = mlflow.get_experiment_by_name(task_name)
        if experiment is None:
            return 0.0

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.best_val_accuracy DESC"])
        if len(runs) == 0:
            return 0.0

        best = runs.iloc[0]
        return best.get("metrics.best_val_accuracy", 0.0)
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="MLOps 评估脚本")
    parser.add_argument("--task", type=str, required=True,
                        help="任务名，对应 tasks/ 下的目录名")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="全局配置文件路径")
    args = parser.parse_args()

    # ---------- 1. 加载配置（全局 + 任务级合并） ----------
    config = load_merged_config(args.config, args.task)
    config["_task_name"] = args.task

    # ---------- 2. 导入任务模块 ----------
    try:
        importlib.import_module(f"tasks.{args.task}.model")
        importlib.import_module(f"tasks.{args.task}.dataset")
        task_pkg = importlib.import_module(f"tasks.{args.task}")
        task_default_dataset = getattr(task_pkg, "DEFAULT_DATASET", "")
        task_default_model = getattr(task_pkg, "DEFAULT_MODEL", "")
    except ModuleNotFoundError as e:
        print(f"错误: 找不到任务模块 'tasks.{args.task}'")
        return
        task_default_dataset = ""
        task_default_model = ""

    # ---------- 3. 确定输出目录和模型路径 ----------
    output_dir = os.path.join(
        config.get("paths", {}).get("output_dir", "./outputs"), args.task)
    best_model_path = os.path.join(output_dir, "best_model.pth")

    if not os.path.exists(best_model_path):
        print(f"错误: 找不到模型文件 {best_model_path}")
        print("请先运行 train.py 训练模型")
        return

    # ---------- 4. 创建模型和测试集 ----------
    model_name = config.get("_model_name",
                            config.get("full", {}).get("model_name", "simple_cnn"))
    config["_model_name"] = model_name

    # 如果注册表中没有这个模型名，就用第一个注册的
    if model_name not in MODELS:
        available = list(MODELS.keys())
        if not available:
            print("错误: 没有注册任何模型")
            return
        model_name = available[0]
        config["_model_name"] = model_name

    model = MODELS[model_name].from_config(config)

    dataset_name = config.get("_dataset_name") or task_default_dataset or list(DATASETS.keys())[0]
    config["_dataset_name"] = dataset_name

    test_dataset = DATASETS[dataset_name].from_config(config, split="test")
    config["_num_classes"] = test_dataset.num_classes

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=0, pin_memory=True)

    # ---------- 5. 加载模型权重 ----------
    device = get_device(config)
    info = load_checkpoint(model, best_model_path, device)
    model.to(device)
    model.eval()

    task_type = getattr(model, "task_type", "classification")

    print(f"{'='*60}")
    print(f"评估任务: {args.task}  |  类型: {task_type}")
    print(f"模型: {model_name}  |  设备: {device}")
    print(f"测试集大小: {len(test_dataset)}")
    if info["epoch"] >= 0:
        best_key = "loss" if task_type == "segmentation" else "accuracy"
        best_val = info['metrics'].get(best_key,
                     info['metrics'].get('accuracy',
                     info['metrics'].get('loss', 0)))
        print(f"加载的模型来自 Epoch {info['epoch']}，"
              f"val_{best_key}={best_val:.4f}")
    print(f"{'='*60}")

    # ---------- 6. 在测试集上评估 ----------
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    # 分割指标
    seg_intersection = 0.0
    seg_union = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if task_type == "classification":
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
            elif task_type == "segmentation":
                pred = (torch.sigmoid(outputs) > 0.5).float()
                # 对逐帧输出取平均或选中间帧
                if pred.dim() == 5:
                    pred = pred[:, :, pred.shape[2] // 2]  # 取中间帧
                if targets.dim() == pred.dim() - 1:
                    targets = targets.unsqueeze(1)
                seg_intersection += (pred * targets).sum().item()
                seg_union += (pred + targets).clamp(0, 1).sum().item()

    class_names = getattr(test_dataset, "CLASS_NAMES",
                          [str(i) for i in range(test_dataset.num_classes)])
    if task_type == "classification":
        accuracy = correct / total if total > 0 else 0.0
        print(f"\n测试准确率: {accuracy:.4f} ({correct}/{total})")
        print("\n分类报告:")
        print(classification_report(
            all_labels, all_preds,
            labels=range(min(test_dataset.num_classes, len(class_names))),
            target_names=class_names[:test_dataset.num_classes],
            zero_division=0))
    elif task_type == "segmentation":
        iou = seg_intersection / (seg_union + 1e-7)
        print(f"\n测试 IoU: {iou:.4f}")
        accuracy = iou  # 用于后续对比

    # ---------- 8. 对比历史最佳 ----------
    mlflow_uri = config.get("paths", {}).get("mlflow_uri", "./mlruns")
    best_historical = get_best_historical_accuracy(args.task, mlflow_uri)
    metric_name = "IoU" if task_type == "segmentation" else "准确率"
    print(f"\n历史最佳{metric_name}: {best_historical:.4f}")
    print(f"当前模型{metric_name}: {accuracy:.4f}")

    if accuracy >= best_historical > 0:
        print(f"结果: 当前模型达到/超越历史最佳!")
    elif best_historical == 0:
        print("结果: 这是本任务的第一次评估")
    else:
        print("结果: 当前模型未超越历史最佳")

    # ---------- 9. 生成图表 ----------
    plot_comparison(accuracy, best_historical, args.task, output_dir)
    if task_type == "classification" and all_labels:
        plot_confusion_matrix(all_labels, all_preds, class_names,
                              args.task, output_dir)

    print(f"\n所有结果保存在: {output_dir}/")


if __name__ == "__main__":
    main()
