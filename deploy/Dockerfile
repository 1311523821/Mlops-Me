# MLOps 推理服务 Docker 镜像
# 使用 CPU 版 onnxruntime，适配 Jetson、树莓派、晟腾、瑞芯微等边缘设备
#
# 构建:
#   docker build -t mlops-inference -f deploy/Dockerfile .
#
# 运行:
#   docker run -p 8000:8000 -v $(pwd)/outputs:/app/outputs mlops-inference --model /app/outputs/demo/model.onnx

FROM python:3.10-slim

# 避免 pip 缓存，减小镜像体积
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# 安装 ONNX Runtime 和 FastAPI 相关依赖
# 这里用 CPU 版 onnxruntime，边缘设备通常用 CPU 推理
RUN pip install --no-cache-dir \
    onnxruntime>=1.15.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    python-multipart>=0.0.6 \
    pillow>=10.0.0 \
    numpy>=1.24.0 \
    onnx>=1.14.0

# 复制推理服务和模型
COPY deploy/app.py /app/app.py

# 暴露端口
EXPOSE 8000

# 启动命令：python app.py --model <模型路径>
# 模型路径通过 volume 挂载或构建时复制
ENTRYPOINT ["python", "app.py"]
CMD ["--model", "/app/outputs/demo/model.onnx"]
