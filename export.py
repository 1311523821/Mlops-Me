"""
ONNX 导出脚本。

用法:
    python export.py --task demo

执行流程:
    1. 加载 best_model.pth
    2. 获取示例输入
    3. torch.onnx.export 导出 ONNX 模型
    4. onnxruntime 加载 ONNX 并推理
    5. 对比 PyTorch 和 ONNX 的输出，验证精度
    6. 保存 model.onnx 到 outputs/<task>/
"""

import argparse
import importlib
import json
import os
import numpy as np
import torch
import onnx
import onnxruntime as ort

from core.utils import load_merged_config, get_device, load_checkpoint
from core.registry import MODELS, DATASETS


def validate_onnx_output(pytorch_output: np.ndarray,
                          onnx_output: np.ndarray) -> float:
    """
    对比 PyTorch 和 ONNX 输出的最大差异。

    参数:
        pytorch_output: PyTorch 模型的输出
        onnx_output:    ONNX 模型的输出

    返回:
        float: 最大绝对差异。接近 0 说明导出精度无损。
    """
    diff = np.abs(pytorch_output - onnx_output)
    return float(diff.max())


def main():
    parser = argparse.ArgumentParser(description="MLOps ONNX 导出脚本")
    parser.add_argument("--task", type=str, required=True,
                        help="任务名，对应 tasks/ 下的目录名")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="全局配置文件路径")
    parser.add_argument("--opset", type=int, default=None,
                        help="ONNX opset 版本（默认使用 config.yaml 中的值）")
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
    except ModuleNotFoundError as e:
        print(f"错误: 找不到任务模块 'tasks.{args.task}'")
        return

    # ---------- 3. 确定路径 ----------
    output_dir = os.path.join(
        config.get("paths", {}).get("output_dir", "./outputs"), args.task)
    best_model_path = os.path.join(output_dir, "best_model.pth")
    onnx_model_path = os.path.join(output_dir, "model.onnx")

    if not os.path.exists(best_model_path):
        print(f"错误: 找不到模型文件 {best_model_path}")
        print("请先运行 train.py 训练模型")
        return

    # ---------- 4. 创建模型并加载权重 ----------
    model_name = config.get("_model_name",
                            config.get("full", {}).get("model_name", "simple_cnn"))
    config["_model_name"] = model_name

    if model_name not in MODELS:
        available = list(MODELS.keys())
        if not available:
            print("错误: 没有注册任何模型")
            return
        model_name = available[0]
        config["_model_name"] = model_name

    model = MODELS[model_name].from_config(config)
    device = get_device(config)
    load_checkpoint(model, best_model_path, device)
    model.to(device)
    model.eval()

    print(f"{'='*60}")
    print(f"ONNX 导出任务: {args.task}")
    print(f"模型: {model_name}  |  设备: {device}")
    print(f"{'='*60}")

    # ---------- 5. 获取示例输入 ----------
    example_input = model.get_example_input().to(device)
    input_names = ["input"]
    output_names = ["output"]

    # 获取 ONNX opset 版本
    opset_version = args.opset or config.get("export", {}).get("onnx_opset", 14)
    print(f"ONNX opset 版本: {opset_version}")
    print(f"示例输入形状: {list(example_input.shape)}")

    # ---------- 6. 导出 ONNX ----------
    os.makedirs(output_dir, exist_ok=True)
    torch.onnx.export(
        model,
        example_input,
        onnx_model_path,
        export_params=True,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX 模型已导出: {onnx_model_path}")

    # ---------- 7. 验证 ONNX 模型 ----------
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型结构验证: 通过")

    # ---------- 8. 对比 PyTorch 和 ONNX 输出 ----------
    example_input_np = example_input.cpu().numpy()

    # PyTorch 推理
    with torch.no_grad():
        pytorch_output = model(example_input).cpu().numpy()

    # ONNX Runtime 推理
    ort_session = ort.InferenceSession(onnx_model_path,
                                       providers=["CPUExecutionProvider"])
    onnx_output = ort_session.run(
        output_names, {input_names[0]: example_input_np})[0]

    max_diff = validate_onnx_output(pytorch_output, onnx_output)
    print(f"\nPyTorch vs ONNX 输出最大差异: {max_diff:.8f}")
    if max_diff < 1e-5:
        print("结论: 导出精度无损，ONNX 模型可以放心使用")
    elif max_diff < 1e-3:
        print("结论: 差异很小（< 0.001），ONNX 模型可用")
    else:
        print("警告: 差异较大，请检查模型和导出设置")

    # ---------- 9. 导出预处理配置（供 deploy/app.py 使用） ----------
    dataset_name = config.get("_dataset_name") or task_default_dataset or list(DATASETS.keys())[0]
    if dataset_name in DATASETS:
        temp_dataset = DATASETS[dataset_name].from_config(config, split="test")
        preprocess = temp_dataset.get_preprocess_config()
        preprocess_path = os.path.join(output_dir, "preprocess.json")
        with open(preprocess_path, "w", encoding="utf-8") as f:
            json.dump(preprocess, f, ensure_ascii=False, indent=2)
        print(f"预处理配置已导出: {preprocess_path}")

    # ---------- 10. 打印模型信息 ----------
    file_size_mb = os.path.getsize(onnx_model_path) / (1024 * 1024)
    print(f"\nONNX 模型文件大小: {file_size_mb:.2f} MB")
    print(f"ONNX 模型保存在: {onnx_model_path}")
    print(f"\n使用方式:")
    print(f"  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{onnx_model_path}')")
    print(f"  result = session.run(['output'], {{'input': your_input}})")


if __name__ == "__main__":
    main()
