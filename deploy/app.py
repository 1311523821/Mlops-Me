"""
FastAPI 推理服务。

启动方式:
    python deploy/app.py --model outputs/demo/best_model.pth

接口:
    GET  /health              → 健康检查
    POST /predict             → 上传图片文件推理
    GET  /predict?url=xxx     → 通过图片 URL 推理
"""

import argparse
import json
import os
import sys
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
from torchvision import transforms
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
import ipaddress
import socket
from urllib.parse import urlparse
import urllib.request


def _is_safe_url(url: str) -> bool:
    """
    SSRF 防护：校验 URL 是否安全，阻止访问内网地址。

    检查项：
    1. 仅允许 http / https scheme
    2. hostname 不能为空
    3. 解析后的 IP 不能是私有地址、回环地址、链路本地地址或未指定地址
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    # 仅允许 http 和 https
    if parsed.scheme not in ("http", "https"):
        return False

    hostname = parsed.hostname
    if hostname is None:
        return False

    # 解析域名到 IP 地址
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False

    resolved_ips = {info[4][0] for info in addr_info}

    for ip_str in resolved_ips:
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False

        # 阻止所有非公网地址
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_unspecified:
            return False

        # 额外检查：阻止特殊的回环段 127.0.0.0/8（ipaddress 已覆盖）
        if ip.version == 6 and ip.is_multicast:
            return False

    return True


# 全局变量，启动时初始化
SESSION = None
TRANSFORM = None
CLASS_NAMES = None
DEVICE = "cpu"


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    app = FastAPI(
        title="MLOps 推理服务",
        description="基于 ONNX Runtime 的深度学习模型推理服务",
        version="1.0.0",
    )

    @app.get("/health")
    async def health():
        """健康检查接口"""
        return {
            "status": "ok",
            "device": DEVICE,
            "model_loaded": SESSION is not None,
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        """
        上传图片文件进行推理。

        curl -X POST -F "file=@digit.png" http://127.0.0.1:8000/predict
        """
        if SESSION is None:
            raise HTTPException(status_code=503, detail="模型未加载")

        # 读取并预处理图片
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        input_tensor = TRANSFORM(image).unsqueeze(0).numpy()

        # ONNX 推理
        outputs = SESSION.run(["output"], {"input": input_tensor})
        probabilities = softmax(outputs[0][0])

        # 构建结果
        results = []
        for idx in np.argsort(-probabilities)[:5]:
            name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
            results.append({
                "class_id": int(idx),
                "class_name": name,
                "probability": float(probabilities[idx]),
            })

        return {
            "filename": file.filename,
            "prediction": results[0],
            "top5": results,
        }

    @app.get("/predict")
    async def predict_url(url: str = Query(..., description="图片 URL")):
        """
        通过图片 URL 进行推理。

        http://127.0.0.1:8000/predict?url=https://example.com/digit.png
        """
        if SESSION is None:
            raise HTTPException(status_code=503, detail="模型未加载")

        # SSRF 防护：校验 URL 安全性
        if not _is_safe_url(url):
            raise HTTPException(status_code=400, detail="不允许访问内网地址或无效的 URL")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MLOps/1.0"})
            image_bytes = urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法下载图片: {e}")

        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        input_tensor = TRANSFORM(image).unsqueeze(0).numpy()

        outputs = SESSION.run(["output"], {"input": input_tensor})
        probabilities = softmax(outputs[0][0])

        results = []
        for idx in np.argsort(-probabilities)[:5]:
            name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
            results.append({
                "class_id": int(idx),
                "class_name": name,
                "probability": float(probabilities[idx]),
            })

        return {
            "url": url,
            "prediction": results[0],
            "top5": results,
        }

    return app


def softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax 计算"""
    x = x - x.max()
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def main():
    global SESSION, TRANSFORM, CLASS_NAMES, DEVICE

    parser = argparse.ArgumentParser(description="MLOps 推理服务")
    parser.add_argument("--model", type=str, required=True,
                        help="ONNX 模型文件路径")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="监听地址")
    parser.add_argument("--port", type=int, default=8000,
                        help="监听端口")
    parser.add_argument("--num-classes", type=int, default=10,
                        help="类别总数")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"错误: 找不到 ONNX 模型文件 {args.model}")
        print("请先运行 python export.py --task <任务名>")
        sys.exit(1)

    # 初始化 ONNX Runtime 会话
    # 尝试用 GPU，不可用则回退 CPU
    providers = ["CPUExecutionProvider"]
    try:
        import onnxruntime.transformers as _  # noqa
    except ImportError:
        pass

    SESSION = ort.InferenceSession(args.model, providers=providers)
    DEVICE = "cpu"

    # 从 ONNX 同目录读取预处理配置（export.py 导出时自动生成）
    model_dir = os.path.dirname(os.path.abspath(args.model))
    preprocess_path = os.path.join(model_dir, "preprocess.json")
    if os.path.exists(preprocess_path):
        with open(preprocess_path, "r", encoding="utf-8") as f:
            preprocess = json.load(f)
        print(f"从 {preprocess_path} 加载预处理配置")
    else:
        # 兜底：从 ONNX 输入形状推断
        input_shape = SESSION.get_inputs()[0].shape
        img_size = input_shape[2] if len(input_shape) >= 3 else 28
        channels = input_shape[1] if len(input_shape) >= 2 else 1
        preprocess = {
            "mean": [0.5] * channels,
            "std": [0.5] * channels,
            "size": [img_size, img_size],
            "channels": channels,
            "classes": [str(i) for i in range(args.num_classes)],
        }
        print("未找到 preprocess.json，使用默认预处理")

    mean = preprocess["mean"]
    std = preprocess["std"]
    img_size = preprocess["size"][0]
    channels = preprocess["channels"]

    print(f"图像尺寸: {img_size}×{img_size}, 通道数: {channels}")
    print(f"归一化: mean={mean}, std={std}")

    # 根据配置动态构建 transform，不硬编码任何任务特定参数
    TRANSFORM = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    CLASS_NAMES = preprocess.get("classes", [str(i) for i in range(args.num_classes)])

    print(f"ONNX 模型已加载: {args.model}")
    print(f"推理设备: {DEVICE}")
    print(f"服务启动: http://{args.host}:{args.port}")

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
