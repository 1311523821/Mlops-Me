"""Paper2MLOps Harness 工具层 — 平台无关的论文/代码分析工具包。

提供：
- harvester: 输入获取（arXiv 下载、git clone、网页抓取）
- parser: 论文解析（PDF 文本提取、架构段落定位）
- analyzer: 代码分析（AST 类签名提取、依赖图、forward 追踪）
- reports: 报告生成（JSON Schema + HTML 渲染）
- server: 本地 HTTP 服务器（报告交付 + 用户确认事件回传）
- generator: MLOps 任务代码生成（model.py/dataset.py/config.yaml）
- validator: 验证（导入链/前向传播/train --fast）
"""
