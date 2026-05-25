"""
训练器：封装标准训练循环，单 GPU 训练，自动 MLflow 记录。

每次训练自动记录：
- 项目名称和标签（config.yaml project 段）
- 数据指纹（数据集 MD5 哈希，追踪数据版本）
- Git commit（代码版本回溯）
- 所有超参数和每 epoch 指标
"""

import os
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import get_device, save_checkpoint


class Trainer:
    """
    标准训练器。

    封装了完整的训练流程：
    - 自动选择设备（GPU/CPU）
    - 训练循环（train_epoch → validate）
    - Early stopping
    - MLflow 记录参数、指标、数据指纹、项目标签
    - 保存最佳模型

    用法:
        trainer = Trainer(model, train_loader, val_loader, config)
        best_metrics = trainer.train()
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: dict,
                 resume_from: str = None):
        """
        参数:
            model:        模型实例
            train_loader: 训练集 DataLoader
            val_loader:   验证集 DataLoader
            config:       完整配置字典
            resume_from:  断点续训的 checkpoint 路径，None 表示从头训练
        """
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

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 输出路径
        task_name = config.get("_task_name", "default")
        self.output_dir = os.path.join(
            config.get("paths", {}).get("output_dir", "./outputs"), task_name)
        self.best_model_path = os.path.join(self.output_dir, "best_model.pth")

        # 断点续训：加载 checkpoint，恢复 epoch 和最佳指标
        self.start_epoch = 1
        if resume_from and os.path.exists(resume_from):
            from .utils import load_checkpoint
            info = load_checkpoint(model, resume_from, self.device)
            self.start_epoch = info["epoch"] + 1
            self.best_val_accuracy = info["metrics"].get("accuracy", 0.0)
            # 恢复 optimizer 状态
            checkpoint = torch.load(resume_from, map_location=self.device,
                                    weights_only=False)
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"从 {resume_from} 恢复训练 (epoch {self.start_epoch}, "
                  f"best_val_acc={self.best_val_accuracy:.4f})")
        else:
            self.best_val_accuracy = 0.0
        self.patience_counter = 0

    def train(self) -> dict:
        """执行完整训练流程。"""
        # ---- MLflow 设置 ----
        mlflow_uri = self.config.get("paths", {}).get("mlflow_uri", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(mlflow_uri)

        # 用 project.name 作为实验名（同一项目不同 task 共用实验，通过 tag 区分）
        project_cfg = self.config.get("project", {})
        project_name = project_cfg.get("name", self.config.get("_task_name", "default"))
        mlflow.set_experiment(project_name)

        # Run 名称：task-模式-时间戳，方便在 MLflow UI 里一眼认出
        run_name = self.config.get("_run_name", f"{self.config.get('_task_name', 'run')}")
        tags = {
            "task": self.config.get("_task_name", "unknown"),
            "mode": "fast" if self.config.get("_data_fraction", 1.0) < 1.0 else "full",
            "data_fingerprint": self.config.get("_data_fingerprint", "unknown"),
            "git_hash": self.config.get("_git_hash", "unknown"),
        }
        # 合并 config.yaml 中 project.tags
        user_tags = project_cfg.get("tags", {})
        tags.update(user_tags)

        with mlflow.start_run(run_name=run_name, tags=tags):
            # ---- 记录参数 ----
            mlflow.log_params({
                "task_name": self.config.get("_task_name", "unknown"),
                "model_name": self.config.get("_model_name", "unknown"),
                "dataset_name": self.config.get("_dataset_name", "unknown"),
                "num_classes": self.config.get("_num_classes", 0),
                "epochs": self.epochs,
                "learning_rate": self.lr,
                "batch_size": self.train_loader.batch_size,
                "optimizer": type(self.optimizer).__name__,
                "device": str(self.device),
            })

            for epoch in range(self.start_epoch, self.epochs + 1):
                train_loss = self._train_epoch(epoch)
                val_metrics = self._validate()
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["accuracy"]

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }, step=epoch)

                print(f"Epoch {epoch:3d}/{self.epochs} | "
                      f"train_loss: {train_loss:.4f} | "
                      f"val_loss: {val_loss:.4f} | "
                      f"val_acc: {val_acc:.4f}")

                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.patience_counter = 0
                    save_checkpoint(self.model, self.optimizer, epoch,
                                    val_metrics, self.best_model_path)
                    print(f"  -> 保存最佳模型 (val_acc={val_acc:.4f})")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"  -> Early stopping, 最佳 val_acc="
                              f"{self.best_val_accuracy:.4f}")
                        break

            mlflow.log_metric("best_val_accuracy", self.best_val_accuracy)
            if os.path.exists(self.best_model_path):
                mlflow.log_artifact(self.best_model_path)

            print(f"\n训练完成! 最佳 val_accuracy: {self.best_val_accuracy:.4f}")
            print(f"模型保存在: {self.best_model_path}")

        return {"val_accuracy": self.best_val_accuracy}

    def _train_epoch(self, epoch: int) -> float:
        """一个 epoch 的训练，返回平均 loss。"""
        self.model.train()
        total_loss = 0.0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate(self) -> dict:
        """验证集评估。"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        return {
            "loss": total_loss / len(self.val_loader),
            "accuracy": correct / total if total > 0 else 0.0,
        }
