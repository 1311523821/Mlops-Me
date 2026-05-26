"""
训练器：封装标准训练循环，单 GPU 训练，自动 MLflow 记录。

支持分类和分割任务，通过模型的 task_type 和 get_loss_fn 自动适配，
Trainer 本身不硬编码任何具体任务逻辑。
"""

import os
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import get_device, save_checkpoint


class Trainer:
    """
    通用训练器，不绑定具体任务类型。

    封装了完整的训练流程：
    - 自动选择设备（GPU/CPU）
    - 从模型获取 loss 函数（模型说了算）
    - 训练循环 → 验证
    - Early stopping（以 val_loss 为基准，所有任务通用）
    - MLflow 记录参数、指标、数据指纹、项目标签
    - 保存最佳模型
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: dict,
                 resume_from: str = None):
        self.config = config
        self.device = get_device(config)
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # 训练参数
        train_cfg = config.get("training", {})
        self.epochs = config.get("_epochs", train_cfg.get("epochs", 50))
        self.lr = train_cfg.get("learning_rate", 0.001)
        self.patience = train_cfg.get("early_stopping_patience", 10)

        # 优化器
        opt_name = train_cfg.get("optimizer", "adam").lower()
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=self.lr,
                weight_decay=train_cfg.get("weight_decay", 0.0001))
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=self.lr, momentum=0.9,
                weight_decay=train_cfg.get("weight_decay", 0.0001))
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # 损失函数 —— 由模型提供，Trainer 不关心具体是什么
        self.criterion = model.get_loss_fn()

        # 任务类型 —— 影响验证指标的计算方式
        self.task_type = getattr(model, "task_type", "classification")

        # 输出路径
        task_name = config.get("_task_name", "default")
        self.output_dir = os.path.join(
            config.get("paths", {}).get("output_dir", "./outputs"), task_name)
        self.best_model_path = os.path.join(self.output_dir, "best_model.pth")

        # 断点续训
        self.best_val_loss = float("inf")
        self.start_epoch = 1
        if resume_from and os.path.exists(resume_from):
            from .utils import load_checkpoint
            info = load_checkpoint(model, resume_from, self.device)
            self.start_epoch = info["epoch"] + 1
            self.best_val_loss = info["metrics"].get("loss", float("inf"))
            checkpoint = torch.load(resume_from, map_location=self.device,
                                    weights_only=False)
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"从 {resume_from} 恢复训练 (epoch {self.start_epoch}, "
                  f"best_val_loss={self.best_val_loss:.4f})")
        self.patience_counter = 0

    def train(self) -> dict:
        """执行完整训练流程。"""
        # ---- MLflow 设置 ----
        mlflow_uri = self.config.get("paths", {}).get("mlflow_uri", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(mlflow_uri)

        project_cfg = self.config.get("project", {})
        project_name = project_cfg.get("name", self.config.get("_task_name", "default"))
        mlflow.set_experiment(project_name)

        run_name = self.config.get("_run_name", f"{self.config.get('_task_name', 'run')}")
        tags = {
            "task": self.config.get("_task_name", "unknown"),
            "mode": "fast" if self.config.get("_data_fraction", 1.0) < 1.0 else "full",
            "task_type": self.task_type,
            "data_fingerprint": self.config.get("_data_fingerprint", "unknown"),
            "git_hash": self.config.get("_git_hash", "unknown"),
        }
        user_tags = project_cfg.get("tags", {})
        tags.update(user_tags)

        with mlflow.start_run(run_name=run_name, tags=tags):
            mlflow.log_params({
                "task_name": self.config.get("_task_name", "unknown"),
                "task_type": self.task_type,
                "model_name": self.config.get("_model_name", "unknown"),
                "dataset_name": self.config.get("_dataset_name", "unknown"),
                "epochs": self.epochs,
                "learning_rate": self.lr,
                "batch_size": self.train_loader.batch_size,
                "optimizer": type(self.optimizer).__name__,
                "loss_function": type(self.criterion).__name__,
                "device": str(self.device),
            })

            for epoch in range(self.start_epoch, self.epochs + 1):
                train_loss = self._train_epoch(epoch)
                val_metrics = self._validate()
                val_loss = val_metrics["loss"]

                # 记录指标（val_loss 是所有任务都有的，额外指标按任务类型追加）
                log_dict = {"train_loss": train_loss, "val_loss": val_loss}
                for k, v in val_metrics.items():
                    if k != "loss":
                        log_dict[f"val_{k}"] = v
                mlflow.log_metrics(log_dict, step=epoch)

                # 打印
                extra_str = " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()
                                     if k != "loss")
                print(f"Epoch {epoch:3d}/{self.epochs} | "
                      f"train_loss: {train_loss:.4f} | "
                      f"val_loss: {val_loss:.4f} | "
                      f"{extra_str}")

                # Early stopping —— 以 val_loss 为基准（所有任务通用）
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    save_checkpoint(self.model, self.optimizer, epoch,
                                    val_metrics, self.best_model_path)
                    print(f"  -> 保存最佳模型 (val_loss={val_loss:.4f})")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"  -> Early stopping, 最佳 val_loss="
                              f"{self.best_val_loss:.4f}")
                        break

            mlflow.log_metric("best_val_loss", self.best_val_loss)
            if os.path.exists(self.best_model_path):
                mlflow.log_artifact(self.best_model_path)

            print(f"\n训练完成! 最佳 val_loss: {self.best_val_loss:.4f}")
            print(f"模型保存在: {self.best_model_path}")

        return {"best_val_loss": self.best_val_loss}

    def _train_epoch(self, epoch: int) -> float:
        """一个 epoch 的训练，返回平均 loss。"""
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            inputs, targets = self._to_device(batch)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate(self) -> dict:
        """
        验证集评估。
        根据 task_type 计算不同的指标：
        - classification: accuracy
        - segmentation: iou
        始终返回 loss。
        """
        self.model.eval()
        total_loss = 0.0

        # 分类指标
        correct = 0
        total = 0
        # 分割指标
        intersection = 0.0
        union = 0.0

        for batch in self.val_loader:
            inputs, targets = self._to_device(batch)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()

            if self.task_type == "classification":
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
            elif self.task_type == "segmentation":
                pred = (torch.sigmoid(outputs) > 0.5).float()
                # 兼容 (B,C,H,W) 和 (B,H,W) 两种标签
                if targets.dim() == pred.dim() - 1:
                    targets = targets.unsqueeze(1)
                intersection += (pred * targets).sum().item()
                union += (pred + targets).clamp(0, 1).sum().item()

        metrics = {"loss": total_loss / len(self.val_loader)}
        if self.task_type == "classification":
            metrics["accuracy"] = correct / total if total > 0 else 0.0
        elif self.task_type == "segmentation":
            iou = intersection / (union + 1e-7)
            metrics["iou"] = iou
        return metrics

    def _to_device(self, batch):
        """将 batch 中的张量移到设备上。兼容 2 元组和更长元组。"""
        inputs, targets = batch[0], batch[1]
        return inputs.to(self.device), targets.to(self.device)
